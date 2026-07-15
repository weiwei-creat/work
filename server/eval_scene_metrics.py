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
Offline scene-quality metrics for SAGE-generated layouts.

Inspired by SceneEval-style metrics (SceneSmith), computed purely from the
generated layout geometry + the shared occupancy grid — no Isaac/physics launch
needed, so it is cheap to run on any generated layout for regression tracking.

Metrics (per room):
  CNT  - object counts (total / floor / on-surface / wall)
  COL  - collision/overlap rate: fraction of object pairs whose footprints
         overlap (checked among floor objects, and among objects sharing the
         same supporter)
  OOB  - out-of-bounds rate: fraction of objects whose footprint exceeds the
         room rectangle
  NAV  - navigability: largest connected free-floor fraction after inflating
         obstacles by a robot foot-print radius (1.0 == the whole free area is
         one connected region a robot can traverse)
  ACC  - accessibility: fraction of floor objects a robot foot-print can stand
         next to (reachable)
  STB* - stability proxy: fraction of on-surface objects whose footprint is well
         supported (centre over the supporter, no large overhang). Proxy only —
         true stability needs a physics settle.

Usage:
  conda activate sage            # NOT sage5080 (matplotlib conflict)
  cd server
  python eval_scene_metrics.py --layout_id <layout_id> [--room_id <id>] [--robot_radius 0.3]
  python eval_scene_metrics.py --layout_id <layout_id> --json_out /tmp/metrics.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import RESULTS_DIR
from objects.object_mobile_manipulation_utils import (
    create_unified_occupancy_grid,
    CollisionCheckingConfig,
)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _aabb_of_rotated_rect(cx, cy, width, length, yaw_rad):
    """Axis-aligned bbox half-extents of a rotated rectangle (width=x, length=y)."""
    hw, hl = width / 2.0, length / 2.0
    c, s = abs(np.cos(yaw_rad)), abs(np.sin(yaw_rad))
    ex = hw * c + hl * s
    ey = hw * s + hl * c
    return (cx - ex, cy - ey, cx + ex, cy + ey)


def _aabb_overlap_area(a, b):
    """Overlap area of two AABBs (x0,y0,x1,y1)."""
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy


def _obj_yaw(obj):
    try:
        return np.deg2rad(float(obj.rotation.z))
    except Exception:
        return 0.0


def _obj_aabb(obj):
    return _aabb_of_rotated_rect(
        obj.position.x, obj.position.y,
        obj.dimensions.width, obj.dimensions.length, _obj_yaw(obj),
    )


def _footprint_area(obj):
    return max(1e-6, obj.dimensions.width * obj.dimensions.length)


# --------------------------------------------------------------------------- #
# Occupancy helpers (mirrors correct_g1_navigation)
# --------------------------------------------------------------------------- #
def _dilate_disk(grid, r_cells):
    if r_cells <= 0:
        return grid.copy()
    try:
        from scipy import ndimage
        yy, xx = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
        disk = (xx * xx + yy * yy) <= r_cells * r_cells
        return ndimage.binary_dilation(grid, structure=disk)
    except Exception:
        out = grid.copy()
        for _ in range(r_cells):
            sh = out.copy()
            sh[1:, :] |= out[:-1, :]; sh[:-1, :] |= out[1:, :]
            sh[:, 1:] |= out[:, :-1]; sh[:, :-1] |= out[:, 1:]
            out = sh
        return out


def _largest_cc_fraction(free):
    """largest connected component / total free cells (8-connectivity)."""
    total = int(free.sum())
    if total == 0:
        return 0.0, 0
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(free, structure=np.ones((3, 3), dtype=int))
        if n == 0:
            return 0.0, 0
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0  # background
        return float(sizes.max()) / total, n
    except Exception:
        return 1.0, 1  # scipy missing: degrade gracefully


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def compute_room_metrics(scene_dir, layout_id, room_id, robot_radius):
    res = CollisionCheckingConfig.GRID_RES

    (occ, gx, gy, bounds, _mesh, floor_plan, room, idx) = create_unified_occupancy_grid(
        scene_dir, layout_id, room_id, only_floor=True, return_idx=True
    )
    n_x, n_y = occ.shape
    rmnx, rmny, rmxx, rmxy = bounds

    objs = list(room.objects)
    floor_objs = [o for o in objs if getattr(o, "place_id", None) == "floor"]
    wall_objs = [o for o in objs if getattr(o, "place_id", None) == "wall"]
    surf_objs = [o for o in objs if getattr(o, "place_id", None) not in ("floor", "wall", None)]

    # ---- CNT ----
    cnt = {
        "total": len(objs),
        "floor": len(floor_objs),
        "on_surface": len(surf_objs),
        "wall": len(wall_objs),
    }

    # ---- COL: footprint overlaps.
    # Reported separately because the semantics differ:
    #  * FLOOR furniture overlapping = a real layout collision (should be ~0).
    #  * On-surface clutter is naturally tight; AABB overlap there over-counts
    #    (adjacent books/cups), so it is informational only and uses a looser
    #    threshold.
    def overlap_rate(group, thresh):
        pairs = overlapping = 0
        bad = set()
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs += 1
                ov = _aabb_overlap_area(_obj_aabb(group[i]), _obj_aabb(group[j]))
                frac = ov / min(_footprint_area(group[i]), _footprint_area(group[j]))
                if frac > thresh:
                    overlapping += 1
                    bad.add(group[i].id); bad.add(group[j].id)
        return pairs, overlapping, bad

    # floor furniture: strict
    f_pairs, f_ov, f_bad = overlap_rate(floor_objs, thresh=0.10)
    # surface clutter: loose, grouped per supporter
    by_support = {}
    for o in surf_objs:
        by_support.setdefault(o.place_id, []).append(o)
    s_pairs = s_ov = 0
    s_bad = set()
    for g in by_support.values():
        p, ov, bad = overlap_rate(g, thresh=0.35)
        s_pairs += p; s_ov += ov; s_bad |= bad
    col = {
        "floor_overlap_rate": (f_ov / f_pairs) if f_pairs else 0.0,
        "floor_objects_in_collision": len(f_bad),
        "floor_collision_rate": (len(f_bad) / len(floor_objs)) if floor_objs else 0.0,
        "surface_overlap_rate_informational": (s_ov / s_pairs) if s_pairs else 0.0,
        "object_collision_rate": (len(f_bad) / len(objs)) if objs else 0.0,
        "note": "floor_* is the real collision signal; surface_* over-counts tight clutter (AABB)",
    }

    # ---- OOB ----
    oob_ids = []
    for o in objs:
        if getattr(o, "place_id", None) != "floor":
            continue
        x0, y0, x1, y1 = _obj_aabb(o)
        if x0 < rmnx - 1e-3 or y0 < rmny - 1e-3 or x1 > rmxx + 1e-3 or y1 > rmxy + 1e-3:
            oob_ids.append(o.id)
    oob = {
        "out_of_bounds": len(oob_ids),
        "oob_rate": (len(oob_ids) / len(floor_objs)) if floor_objs else 0.0,
    }

    # ---- NAV + ACC (occupancy based) ----
    r_cells = int(np.ceil(robot_radius / res))
    inflated = _dilate_disk(occ, r_cells)
    free = ~inflated
    edge = int(np.ceil(CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE / res))
    if edge > 0:
        free[:edge, :] = False; free[-edge:, :] = False
        free[:, :edge] = False; free[:, -edge:] = False
    nav_frac, n_components = _largest_cc_fraction(free)
    nav = {
        "navigable_fraction": round(nav_frac, 4),
        "free_components": n_components,
        "robot_radius": robot_radius,
    }

    # ACC: floor object is accessible if a free cell lies within (radius+1) cells
    occ_idx = idx["occupancy_idx_grid"]
    name_map = idx["idx_to_object_name"]
    name_to_k = {v: k for k, v in name_map.items()}
    reach_r = r_cells + 1
    accessible = 0
    acc_checked = 0
    try:
        from scipy import ndimage
        free_dist = ndimage.distance_transform_edt(~free)  # cells to nearest free
    except Exception:
        free_dist = None
    for o in floor_objs:
        k = name_to_k.get(o.id)
        if k is None:
            continue
        acc_checked += 1
        cells = np.argwhere(occ_idx == k)
        if len(cells) == 0:
            continue
        if free_dist is not None:
            if np.any(free_dist[cells[:, 0], cells[:, 1]] <= reach_r):
                accessible += 1
        else:
            accessible += 1
    acc = {
        "accessible_floor_objects": accessible,
        "checked_floor_objects": acc_checked,
        "accessibility_rate": (accessible / acc_checked) if acc_checked else 1.0,
    }

    # ---- STB proxy: on-surface object centre over supporter, overhang small ----
    id_to_obj = {o.id: o for o in objs}
    well_supported = 0
    stb_checked = 0
    for o in surf_objs:
        sup = id_to_obj.get(o.place_id)
        if sup is None:
            continue
        stb_checked += 1
        sb = _obj_aabb(sup)
        # centre of object within supporter footprint?
        center_ok = (sb[0] <= o.position.x <= sb[2]) and (sb[1] <= o.position.y <= sb[3])
        ob = _obj_aabb(o)
        inside = _aabb_overlap_area(ob, sb)
        overhang_ok = inside >= 0.6 * _footprint_area(o)  # >=60% of footprint supported
        if center_ok and overhang_ok:
            well_supported += 1
    stb = {
        "well_supported": well_supported,
        "checked_on_surface": stb_checked,
        "support_rate_proxy": (well_supported / stb_checked) if stb_checked else 1.0,
        "note": "geometric proxy; true stability needs a physics settle",
    }

    return {
        "room_id": room_id,
        "room_type": getattr(room, "room_type", ""),
        "CNT": cnt, "COL": col, "OOB": oob, "NAV": nav, "ACC": acc, "STB": stb,
    }


def main():
    ap = argparse.ArgumentParser(description="Offline scene-quality metrics for SAGE layouts")
    ap.add_argument("--layout_id", required=True)
    ap.add_argument("--room_id", default=None, help="default: all rooms in the layout")
    ap.add_argument("--robot_radius", type=float, default=0.30)
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    scene_dir = os.path.join(args.results_dir, args.layout_id)
    layout_json = os.path.join(scene_dir, f"{args.layout_id}.json")
    if not os.path.exists(layout_json):
        print(f"[eval] layout json not found: {layout_json}", file=sys.stderr)
        sys.exit(1)
    layout = json.load(open(layout_json))
    room_ids = [args.room_id] if args.room_id else [r["id"] for r in layout["rooms"]]

    report = {"layout_id": args.layout_id, "robot_radius": args.robot_radius, "rooms": []}
    for rid in room_ids:
        try:
            m = compute_room_metrics(scene_dir, args.layout_id, rid, args.robot_radius)
            report["rooms"].append(m)
        except Exception as e:
            import traceback
            print(f"[eval] room {rid} failed: {e}", file=sys.stderr)
            traceback.print_exc()
            report["rooms"].append({"room_id": rid, "error": str(e)})

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[eval] wrote {args.json_out}", file=sys.stderr)

    # One-line summary per room
    for m in report["rooms"]:
        if "error" in m:
            print(f"  {m['room_id']}: ERROR {m['error']}", file=sys.stderr)
            continue
        print(
            f"  {m['room_id']} ({m['room_type']}): "
            f"CNT={m['CNT']['total']} (floor {m['CNT']['floor']}, surf {m['CNT']['on_surface']}) "
            f"COL={m['COL']['object_collision_rate']:.1%} "
            f"OOB={m['OOB']['oob_rate']:.1%} "
            f"NAV={m['NAV']['navigable_fraction']:.2f} "
            f"ACC={m['ACC']['accessibility_rate']:.1%} "
            f"STB*={m['STB']['support_rate_proxy']:.1%}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
