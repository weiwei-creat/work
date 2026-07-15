# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Navigation feasibility correction + path planning for the Unitree G1 humanoid.

Unlike the mobile-franka correction (which validates pick/place reachability and
holonomic-base trajectories), the G1 here performs PURE NAVIGATION: it walks
between floor landmarks. So feasibility reduces to:

    "Does a collision-free path exist for a humanoid foot-print (a disk of
     radius ~0.3 m) connecting the navigation goals, given the placed objects?"

This module:
  1. Rasterizes the room into an occupancy grid (reusing the unified grid used by
     the mobile-franka collision checker, so footprints stay consistent).
  2. Inflates obstacles by the humanoid foot-print radius + room-edge margin.
  3. Resolves the ordered navigation goals from the task decomposition.
  4. Runs an 8-connected grid A* between consecutive goals.
  5. If a segment is blocked, removes the offending NON-landmark objects and
     retries (mirroring the mobile-franka "remove blockers" behaviour).
  6. Writes the resulting way-points (world x, y + heading yaw) to
     ``{layout_id}_g1_nav_path.json`` for the downstream kinematic visualization
     to follow, and saves the corrected layout.

The planner is intentionally kinematic (SE(2) foot-print). It produces the
navigation *task* and a reference path; it does NOT run a locomotion policy.
"""

import heapq
import json
import os
import sys

import numpy as np

# Make the server root importable (mirrors correct_mobile_franka.py).
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from constants import RESULTS_DIR
from utils import export_layout_to_json, dict_to_floor_plan
from objects.object_mobile_manipulation_utils import (
    create_unified_occupancy_grid,
    CollisionCheckingConfig,
)

# Humanoid (Unitree G1) navigation foot-print radius in metres. The G1 standing
# foot-print is small (~0.2 m), but we keep a conservative margin so the
# kinematic walk does not clip furniture.
G1_FOOTPRINT_RADIUS = 0.30

# Maximum number of object-removal retries when a segment is blocked.
MAX_CLEAR_ATTEMPTS = 8


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def _world_to_cell(x, y, room_bounds, n_x, n_y):
    """World (x, y) -> integer grid cell (i, j), clamped to grid bounds."""
    room_min_x, room_min_y, _, _ = room_bounds
    res = CollisionCheckingConfig.GRID_RES
    i = int(np.floor((x - room_min_x) / res))
    j = int(np.floor((y - room_min_y) / res))
    i = int(np.clip(i, 0, n_x - 1))
    j = int(np.clip(j, 0, n_y - 1))
    return i, j


def _cell_to_world(i, j, room_bounds):
    """Grid cell (i, j) -> world (x, y) at the cell centre."""
    room_min_x, room_min_y, _, _ = room_bounds
    res = CollisionCheckingConfig.GRID_RES
    x = room_min_x + (i + 0.5) * res
    y = room_min_y + (j + 0.5) * res
    return x, y


def _build_free_grid(occupancy_grid, room_bounds):
    """
    Build a boolean ``free`` grid: True where the humanoid foot-print centre may
    stand. Obstacles are inflated by the foot-print radius and a band along the
    room edge is forbidden (matches CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE).
    """
    res = CollisionCheckingConfig.GRID_RES
    n_x, n_y = occupancy_grid.shape

    # Inflate occupied cells by the foot-print radius.
    radius_cells = int(np.ceil(G1_FOOTPRINT_RADIUS / res))
    inflated = _binary_dilate_disk(occupancy_grid, radius_cells)

    free = ~inflated

    # Forbid a band along the room edges so the foot-print never clips walls.
    edge_cells = int(np.ceil(CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE / res))
    if edge_cells > 0:
        free[:edge_cells, :] = False
        free[-edge_cells:, :] = False
        free[:, :edge_cells] = False
        free[:, -edge_cells:] = False

    return free


def _binary_dilate_disk(grid, radius_cells):
    """
    Dilate a boolean grid by a disk of ``radius_cells``. Uses scipy if available,
    otherwise falls back to a separable max-filter approximation (square kernel).
    """
    if radius_cells <= 0:
        return grid.copy()
    try:
        from scipy import ndimage

        yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        disk = (xx * xx + yy * yy) <= radius_cells * radius_cells
        return ndimage.binary_dilation(grid, structure=disk)
    except Exception as exc:  # pragma: no cover - scipy is normally present
        print(f"[g1_nav] scipy dilation unavailable ({exc}); using square fallback", file=sys.stderr)
        out = grid.copy()
        for _ in range(radius_cells):
            shifted = out.copy()
            shifted[1:, :] |= out[:-1, :]
            shifted[:-1, :] |= out[1:, :]
            shifted[:, 1:] |= out[:, :-1]
            shifted[:, :-1] |= out[:, 1:]
            out = shifted
        return out


def _nearest_free_cell(free, cell):
    """BFS outward from ``cell`` to the closest free cell. Returns (i, j) or None."""
    n_x, n_y = free.shape
    i0, j0 = cell
    if 0 <= i0 < n_x and 0 <= j0 < n_y and free[i0, j0]:
        return (i0, j0)

    from collections import deque

    visited = np.zeros_like(free, dtype=bool)
    q = deque([(i0, j0)])
    visited[i0, j0] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n_x and 0 <= nj < n_y and not visited[ni, nj]:
                visited[ni, nj] = True
                if free[ni, nj]:
                    return (ni, nj)
                q.append((ni, nj))
    return None


def _astar(free, start, goal):
    """8-connected A* on the boolean ``free`` grid. Returns list of cells or None."""
    n_x, n_y = free.shape
    if not (free[start] and free[goal]):
        return None

    def h(c):
        return np.hypot(c[0] - goal[0], c[1] - goal[1])

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    open_heap = [(h(start), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            # Reconstruct path.
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        if cur in closed:
            continue
        closed.add(cur)
        ci, cj = cur
        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n_x and 0 <= nj < n_y) or not free[ni, nj]:
                continue
            # Prevent diagonal corner-cutting through obstacles.
            if di != 0 and dj != 0 and not (free[ci + di, cj] and free[ci, cj + dj]):
                continue
            nxt = (ni, nj)
            if nxt in closed:
                continue
            step = np.hypot(di, dj)
            tentative = g + step
            if tentative < g_score.get(nxt, np.inf):
                came_from[nxt] = cur
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))
    return None


def _object_cells(idx_dict, obj_id):
    """Return the set of grid cells (as an (i, j) bool mask) belonging to obj_id."""
    occ_idx = idx_dict["occupancy_idx_grid"]
    name_map = idx_dict["idx_to_object_name"]
    target_idx = None
    for k, name in name_map.items():
        if name == obj_id:
            target_idx = k
            break
    if target_idx is None:
        return np.zeros_like(occ_idx, dtype=bool)
    return occ_idx == target_idx


# --------------------------------------------------------------------------- #
# Goal resolution
# --------------------------------------------------------------------------- #
def _resolve_navigation_goals(policy_analysis):
    """
    Extract the ordered list of navigation goal object ids from the task
    decomposition, resolving names -> room object ids via the object mapping.
    """
    updated = policy_analysis.get("updated_task_decomposition", [])
    object_mapping = policy_analysis.get("object_mapping", {})

    goals = []
    if updated:
        for task in updated:
            if task.get("action") == "navigate":
                oid = task.get("target_object_id") or task.get("location_object_id")
                if oid:
                    goals.append(oid)
    else:
        for task in policy_analysis.get("task_decomposition", []):
            if task.get("action") == "navigate":
                name = task.get("target_object", "")
                matched = object_mapping.get(name, {}).get("matched_ids", [])
                if matched:
                    goals.append(matched[0])

    # De-duplicate consecutive identical goals (walking to the same spot twice).
    deduped = []
    for g in goals:
        if not deduped or deduped[-1] != g:
            deduped.append(g)
    return deduped


def _densify_path(cells, room_bounds):
    """Convert a cell path to world way-points with heading yaw toward the next point."""
    pts = [_cell_to_world(i, j, room_bounds) for (i, j) in cells]
    waypoints = []
    for k, (x, y) in enumerate(pts):
        if k < len(pts) - 1:
            nx, ny = pts[k + 1]
            yaw = float(np.arctan2(ny - y, nx - x))
        elif waypoints:
            yaw = waypoints[-1]["yaw"]
        else:
            yaw = 0.0
        waypoints.append({"x": float(x), "y": float(y), "yaw": yaw})
    return waypoints


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
async def correct_g1_navigation_standalone(layout, room_id="", temp_json_path=None):
    """
    Validate (and minimally correct) a room for a Unitree G1 navigation task, and
    emit a reference foot-print path.

    Args:
        layout: FloorPlan object (its policy_analysis must already be matched).
        room_id: room to plan within.
        temp_json_path: where to persist the (possibly corrected) layout.

    Returns:
        JSON string describing feasibility, the way-point path, and removed objects.
    """
    current_layout = layout
    policy_analysis = current_layout.policy_analysis
    robot_type = policy_analysis.get("robot_type", "unitree_g1")

    target_room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if target_room is None:
        return json.dumps({"success": False, "error": f"Room with ID '{room_id}' not found"})

    layout_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
    os.makedirs(layout_save_dir, exist_ok=True)
    if temp_json_path is None:
        temp_json_path = os.path.join(layout_save_dir, f"{current_layout.id}.json")
    temp_layout_name = os.path.splitext(os.path.basename(temp_json_path))[0]

    goals = _resolve_navigation_goals(policy_analysis)
    print(f"[g1_nav] navigation goals (ordered): {goals}", file=sys.stderr)
    if len(goals) < 1:
        return json.dumps({
            "success": False,
            "error": "No navigation goals found in task decomposition for unitree_g1.",
        })

    # Landmarks must never be removed when clearing blockers.
    protected_ids = set(goals)
    for req in policy_analysis.get("minimum_required_objects", []):
        protected_ids.update(req.get("matched_object_ids", []) or [])

    removed_objects = []

    for attempt in range(MAX_CLEAR_ATTEMPTS):
        # Persist current working layout so the occupancy grid reflects removals.
        export_layout_to_json(current_layout, temp_json_path)

        (occupancy_grid, grid_x, grid_y, room_bounds,
         _mesh, _fp, _room, idx_dict) = create_unified_occupancy_grid(
            layout_save_dir, temp_layout_name, room_id, only_floor=True, return_idx=True
        )
        n_x, n_y = occupancy_grid.shape
        free = _build_free_grid(occupancy_grid, room_bounds)

        # Resolve each goal to a free approach cell next to its foot-print.
        goal_cells = []
        unreachable_goal = None
        for gid in goals:
            obj_mask = _object_cells(idx_dict, gid)
            if obj_mask.any():
                # Approach cell = free cell nearest to the object's occupied cells.
                gi, gj = _approach_cell(obj_mask, free)
            else:
                # Landmark not on the floor grid (e.g. small/raised) -> use its centroid.
                gobj = next((o for o in target_room.objects if o.id == gid), None)
                if gobj is None:
                    unreachable_goal = gid
                    break
                ci, cj = _world_to_cell(gobj.position.x, gobj.position.y, room_bounds, n_x, n_y)
                approach = _nearest_free_cell(free, (ci, cj))
                gi, gj = approach if approach else (None, None)
            if gi is None:
                unreachable_goal = gid
                break
            goal_cells.append((gi, gj))

        if unreachable_goal is not None:
            print(f"[g1_nav] goal {unreachable_goal} has no free approach cell", file=sys.stderr)

        # Start point: free cell closest to the room centre.
        center_cell = _world_to_cell(
            (room_bounds[0] + room_bounds[2]) / 2.0,
            (room_bounds[1] + room_bounds[3]) / 2.0,
            room_bounds, n_x, n_y,
        )
        start_cell = _nearest_free_cell(free, center_cell)

        if unreachable_goal is None and start_cell is not None and len(goal_cells) == len(goals):
            # Plan start -> goal_0 -> goal_1 -> ...
            full_cells = []
            cur = start_cell
            failed_segment = None
            for seg_idx, gc in enumerate(goal_cells):
                seg = _astar(free, cur, gc)
                if seg is None:
                    failed_segment = (cur, gc, seg_idx)
                    break
                full_cells.extend(seg if not full_cells else seg[1:])
                cur = gc

            if failed_segment is None:
                # Success: emit way-points and the corrected layout.
                waypoints = _densify_path(full_cells, room_bounds)
                nav_path_file = os.path.join(layout_save_dir, f"{current_layout.id}_g1_nav_path.json")
                with open(nav_path_file, "w") as f:
                    json.dump({
                        "layout_id": current_layout.id,
                        "room_id": room_id,
                        "robot_type": robot_type,
                        "footprint_radius": G1_FOOTPRINT_RADIUS,
                        "goal_object_ids": goals,
                        "waypoints": waypoints,
                    }, f, indent=2)
                print(f"[g1_nav] path with {len(waypoints)} way-points -> {nav_path_file}", file=sys.stderr)
                return json.dumps({
                    "success": True,
                    "robot_type": robot_type,
                    "room_id": room_id,
                    "num_goals": len(goals),
                    "num_waypoints": len(waypoints),
                    "nav_path_file": nav_path_file,
                    "removed_objects": removed_objects,
                })

            # Blocked: try to clear a non-protected object on the failing segment.
            cur_cell, goal_cell, seg_idx = failed_segment
            print(f"[g1_nav] segment {seg_idx} blocked; searching for removable blocker", file=sys.stderr)
            blocker = _find_blocker_on_segment(
                cur_cell, goal_cell, occupancy_grid, idx_dict, protected_ids, room_bounds
            )
        else:
            # No valid start/goal cells; try clearing the densest non-protected object.
            blocker = _find_densest_blocker(idx_dict, protected_ids)

        if blocker is None:
            print("[g1_nav] no removable blocker found; navigation infeasible", file=sys.stderr)
            return json.dumps({
                "success": False,
                "error": "No collision-free path exists and no removable (non-landmark) "
                         "blocking object was found. Consider a larger room or fewer "
                         "floor obstacles between the navigation goals.",
                "removed_objects": removed_objects,
            })

        # Remove the blocker (and anything resting on it) and retry.
        n_removed = _remove_object_and_descendants(target_room, blocker)
        removed_objects.append(blocker)
        print(f"[g1_nav] removed blocker '{blocker}' (+{n_removed - 1} descendants); retrying", file=sys.stderr)

    return json.dumps({
        "success": False,
        "error": f"Could not find a collision-free path after {MAX_CLEAR_ATTEMPTS} clearing attempts.",
        "removed_objects": removed_objects,
    })


def _approach_cell(obj_mask, free):
    """Free cell nearest to any True cell in obj_mask. Returns (i, j) or (None, None)."""
    try:
        from scipy import ndimage

        # Distance (in cells) from every cell to the nearest object cell.
        dist = ndimage.distance_transform_edt(~obj_mask)
        masked = np.where(free, dist, np.inf)
        if not np.isfinite(masked).any():
            return None, None
        flat = int(np.argmin(masked))
        return np.unravel_index(flat, masked.shape)
    except Exception:
        # Fallback: BFS from the object centroid.
        ii, jj = np.where(obj_mask)
        if len(ii) == 0:
            return None, None
        c = (int(ii.mean()), int(jj.mean()))
        res = _nearest_free_cell(free, c)
        return res if res else (None, None)


def _find_blocker_on_segment(cur_cell, goal_cell, occupancy_grid, idx_dict, protected_ids, room_bounds):
    """
    Walk the straight line between two cells; return the id of the first
    non-protected object whose foot-print intersects a corridor around the line.
    """
    occ_idx = idx_dict["occupancy_idx_grid"]
    name_map = idx_dict["idx_to_object_name"]
    res = CollisionCheckingConfig.GRID_RES
    corridor = int(np.ceil(G1_FOOTPRINT_RADIUS / res))
    n_x, n_y = occupancy_grid.shape

    (i0, j0), (i1, j1) = cur_cell, goal_cell
    steps = max(abs(i1 - i0), abs(j1 - j0), 1)
    candidates = {}
    for s in range(steps + 1):
        t = s / steps
        ci = int(round(i0 + t * (i1 - i0)))
        cj = int(round(j0 + t * (j1 - j0)))
        for di in range(-corridor, corridor + 1):
            for dj in range(-corridor, corridor + 1):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < n_x and 0 <= nj < n_y and occupancy_grid[ni, nj]:
                    idx = int(occ_idx[ni, nj])
                    if idx >= 0:
                        oid = name_map.get(idx)
                        if oid and oid not in protected_ids:
                            candidates[oid] = candidates.get(oid, 0) + 1
    if not candidates:
        return None
    # Remove the object contributing the most cells along the corridor.
    return max(candidates, key=candidates.get)


def _find_densest_blocker(idx_dict, protected_ids):
    """Fallback: the non-protected object occupying the most floor cells."""
    occ_idx = idx_dict["occupancy_idx_grid"]
    name_map = idx_dict["idx_to_object_name"]
    counts = {}
    for idx, name in name_map.items():
        if name in protected_ids:
            continue
        counts[name] = int(np.count_nonzero(occ_idx == idx))
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return None
    return max(counts, key=counts.get)


def _remove_object_and_descendants(room, obj_id):
    """Remove obj_id and any object (recursively) resting on it. Returns count removed."""
    to_remove = {obj_id}
    changed = True
    while changed:
        changed = False
        for obj in room.objects:
            pid = getattr(obj, "place_id", None)
            if pid in to_remove and obj.id not in to_remove:
                to_remove.add(obj.id)
                changed = True
    before = len(room.objects)
    room.objects = [o for o in room.objects if o.id not in to_remove]
    return before - len(room.objects)
