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
Standalone renderer: a *legged* Unitree G1 kinematically walks the SAGE nav path
through the generated scene, with RTX rendering captured to PNG frames + MP4.

This is a self-contained SimulationApp script (run it with Isaac's python.sh, NOT
the conda env) so it boots its own renderer and does not disturb the headless MCP
kit. It uses only omni.isaac.core + numpy (no IsaacLab, no scipy).

Run on the machine that has Isaac Sim (100.111.153.8):

    export SAGE_LAYOUT_DIR=/home/gaok/coding/sage/server/results/<layout_id>
    export SAGE_LAYOUT_ID=<layout_id>
    # legged G1 (37 dof, with hip/knee) — default points at the Isaac 5.1 S3 asset:
    # export SAGE_G1_USD=https://.../Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd
    # export SAGE_RENDER_HEADLESS=0   # set 0 + DISPLAY=:0 for a live window
    /home/gaok/coding/isaacsim/_build/linux-x86_64/release/python.sh \
        server/isaacsim/isaac.sim.mcp_extension/examples/g1_walk_render.py

Outputs PNG frames (and frames.mp4 if ffmpeg is available) under SAGE_RENDER_OUT
(default /tmp/g1_render).
"""

import os
import json
import math

import numpy as np

# ---- launch the simulator FIRST (before any omni.* import) ---------------- #
HEADLESS = os.environ.get("SAGE_RENDER_HEADLESS", "1") != "0"
try:
    from isaacsim import SimulationApp
except Exception:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": HEADLESS, "enable_cameras": True})

# ---- now the rest -------------------------------------------------------- #
from omni.isaac.core import World, PhysicsContext  # noqa: E402
from omni.isaac.core.articulations import Articulation  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
from omni.isaac.core.prims import XFormPrim  # noqa: E402
from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402
from omni.isaac.core.utils.viewports import set_camera_view  # noqa: E402

LAYOUT_DIR = os.environ["SAGE_LAYOUT_DIR"]
LAYOUT_ID = os.environ["SAGE_LAYOUT_ID"]
USD_DIR = os.path.join(LAYOUT_DIR, f"{LAYOUT_ID}_usd_collection")
NAV_FILE = os.path.join(LAYOUT_DIR, f"{LAYOUT_ID}_g1_nav_path.json")
OUT_DIR = os.environ.get("SAGE_RENDER_OUT", "/tmp/g1_render")
G1_USD = os.environ.get(
    "SAGE_G1_USD",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd",
)
G1_PELVIS_HEIGHT = 0.74
GAIT_HIP_AMP = 0.45
GAIT_KNEE_AMP = 0.40
GAIT_PERIOD = 16
SUBSTEPS = 3  # render frames spent between two consecutive way-points

os.makedirs(OUT_DIR, exist_ok=True)


def euler_deg_to_quat_wxyz(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    # ZYX intrinsic == xyz extrinsic order used by scipy "xyz"
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z])


def yaw_to_quat_wxyz(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def log(msg):
    print(f"[g1_render] {msg}", flush=True)


def main():
    log(f"layout={LAYOUT_ID} headless={HEADLESS}")
    log(f"G1 USD = {G1_USD}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    PhysicsContext().set_physics_dt(1.0 / 60.0)

    # --- load the generated scene ---
    transforms = {}
    tf_path = os.path.join(USD_DIR, "rigid_object_transform_dict.json")
    if os.path.exists(tf_path):
        transforms = json.load(open(tf_path))
    n_scene = 0
    ceiling_paths = []
    for fn in sorted(os.listdir(USD_DIR)):
        if not fn.endswith(".usdz"):
            continue
        name = os.path.splitext(fn)[0]
        pp = f"/World/scene/{name}"
        add_reference_to_stage(usd_path=os.path.join(USD_DIR, fn), prim_path=pp)
        n_scene += 1
        # The ceiling would occlude an interior top view -> hide it.
        if "ceiling" in name.lower():
            ceiling_paths.append(pp)
            continue
        if name.startswith(("floor_", "wall_", "door_", "window_")):
            continue
        tf = transforms.get(name)
        if not tf:
            continue
        p, r = tf["position"], tf["rotation"]
        XFormPrim(prim_path=pp).set_world_pose(
            position=np.array([p["x"], p["y"], p["z"]]),
            orientation=euler_deg_to_quat_wxyz(r["x"], r["y"], r["z"]),
        )
    log(f"loaded {n_scene} scene prims")
    for cp in ceiling_paths:
        try:
            XFormPrim(prim_path=cp).set_visibility(False)
        except Exception as e:
            log(f"could not hide ceiling {cp}: {e}")
    log(f"hid {len(ceiling_paths)} ceiling prim(s)")

    # --- nav path ---
    nav = json.load(open(NAV_FILE))
    waypoints = nav["waypoints"]
    log(f"{len(waypoints)} way-points")

    # --- room bounds (for camera placement) ---
    layout = json.load(open(os.path.join(LAYOUT_DIR, f"{LAYOUT_ID}.json")))
    room = next((r for r in layout["rooms"] if r["id"] == nav.get("room_id")), layout["rooms"][0])
    rx, ry = room["position"]["x"], room["position"]["y"]
    rw, rl = room["dimensions"]["width"], room["dimensions"]["length"]
    rh = room["dimensions"]["height"]
    room_cx, room_cy = rx + rw / 2.0, ry + rl / 2.0

    # --- legged G1 ---
    add_reference_to_stage(usd_path=G1_USD, prim_path="/World/G1")
    wp0 = waypoints[0]
    XFormPrim(prim_path="/World/G1").set_world_pose(
        position=np.array([wp0["x"], wp0["y"], G1_PELVIS_HEIGHT]),
        orientation=yaw_to_quat_wxyz(wp0["yaw"]),
    )

    # --- camera: high oblique from just outside the -y wall, looking down into
    # the (now open-topped) room so the interior + walking G1 are visible.
    # Tunable via SAGE_CAM_* env vars for quick framing tweaks.
    cam_back = float(os.environ.get("SAGE_CAM_BACK", "0.6"))      # metres outside -y wall
    cam_up = float(os.environ.get("SAGE_CAM_UP", "2.2"))         # metres above ceiling height
    cam_target_z = float(os.environ.get("SAGE_CAM_TARGET_Z", "0.5"))
    eye = [room_cx, ry - cam_back, rh + cam_up]
    target = [room_cx, room_cy, cam_target_z]
    log(f"camera eye={eye} target={target}")
    set_camera_view(eye=eye, target=target)

    world.reset()
    world.play()
    for _ in range(20):
        world.step(render=True)

    g1 = Articulation(prim_path="/World/G1", name="G1")
    g1.initialize(world.physics_sim_view)
    ctrl = g1.get_articulation_controller()
    dof_names = list(g1.dof_names or [])
    nj = len(dof_names)
    log(f"G1 dof count = {nj}")
    log(f"dof_names = {dof_names}")

    # gait joints by name
    leg = {}
    for side in ("left", "right"):
        for key in ("hip_pitch", "knee"):
            m = next((i for i, n in enumerate(dof_names)
                      if side in n.lower() and key.split('_')[0] in n.lower()
                      and (key != "hip_pitch" or "pitch" in n.lower())), None)
            if m is not None:
                leg[f"{side}_{key}"] = m
    log(f"gait joints matched = {leg}")
    ctrl.set_gains(kps=[120.0] * nj, kds=[12.0] * nj)

    def gait_pose(phase):
        pose = np.zeros(nj)
        sw = math.sin(phase)
        if "left_hip_pitch" in leg:
            pose[leg["left_hip_pitch"]] = GAIT_HIP_AMP * sw
        if "right_hip_pitch" in leg:
            pose[leg["right_hip_pitch"]] = -GAIT_HIP_AMP * sw
        if "left_knee" in leg:
            pose[leg["left_knee"]] = GAIT_KNEE_AMP * (0.5 - 0.5 * sw)
        if "right_knee" in leg:
            pose[leg["right_knee"]] = GAIT_KNEE_AMP * (0.5 + 0.5 * sw)
        return pose

    # --- capture helper (viewport screenshot) ---
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    vp = get_active_viewport()
    frame_paths = []

    def capture(idx):
        path = os.path.join(OUT_DIR, f"frame_{idx:04d}.png")
        try:
            capture_viewport_to_file(vp, path)
            frame_paths.append(path)
        except Exception as e:
            log(f"capture failed @ {idx}: {e}")

    # --- walk the route ---
    # In windowed mode, loop the walk so you can actually watch it (the window
    # would otherwise flash through one pass and close). Frames are captured only
    # on the first pass to avoid filling the disk. Tunable via SAGE_WALK_LOOPS.
    loops = int(os.environ.get("SAGE_WALK_LOOPS", "1" if HEADLESS else "1000000"))
    log(f"walking + rendering ... (loops={loops})")
    step = 0
    fidx = 0
    for loop_i in range(loops):
        if not simulation_app.is_running():
            break
        capture_enabled = (loop_i == 0)
        for k in range(len(waypoints) - 1):
            a, b = waypoints[k], waypoints[k + 1]
            for s in range(SUBSTEPS):
                t = (s + 1) / SUBSTEPS
                x = a["x"] + t * (b["x"] - a["x"])
                y = a["y"] + t * (b["y"] - a["y"])
                yaw = b["yaw"]
                ctrl.apply_action(ArticulationAction(joint_positions=gait_pose(
                    (step % GAIT_PERIOD) * 2.0 * math.pi / GAIT_PERIOD)))
                g1.set_world_pose(position=np.array([x, y, G1_PELVIS_HEIGHT]),
                                  orientation=yaw_to_quat_wxyz(yaw))
                world.step(render=True)
                if capture_enabled:
                    capture(fidx)
                    fidx += 1
                step += 1
        # settle at the goal before (optionally) looping back to the start
        for _ in range(8):
            ctrl.apply_action(ArticulationAction(joint_positions=np.zeros(nj)))
            world.step(render=True)
        if capture_enabled:
            capture(fidx)
        # teleport back to the start for the next visible loop
        if loops > 1 and loop_i < loops - 1:
            g1.set_world_pose(position=np.array([wp0["x"], wp0["y"], G1_PELVIS_HEIGHT]),
                              orientation=yaw_to_quat_wxyz(wp0["yaw"]))

    # let async captures flush
    for _ in range(30):
        simulation_app.update()
    log(f"captured ~{len(frame_paths)} frames -> {OUT_DIR}")

    # --- encode mp4 if ffmpeg present ---
    mp4 = os.path.join(OUT_DIR, "frames.mp4")
    rc = os.system(
        f"ffmpeg -y -framerate 15 -pattern_type glob -i '{OUT_DIR}/frame_*.png' "
        f"-c:v libx264 -pix_fmt yuv420p {mp4} >/dev/null 2>&1"
    )
    if rc == 0 and os.path.exists(mp4):
        log(f"wrote {mp4}")
    else:
        log("ffmpeg not available or failed; PNG frames are in OUT_DIR")
    log("DONE")


try:
    main()
except Exception as exc:
    import traceback
    log(f"ERROR: {exc}")
    traceback.print_exc()
finally:
    simulation_app.close()
