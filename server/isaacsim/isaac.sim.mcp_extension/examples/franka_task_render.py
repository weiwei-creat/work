"""Kinematic Franka pick-and-place visualization for a generated SAGE scene.

Runs on plain Isaac Sim 5.x (python.sh + SimulationApp + Lula IK) — it does NOT
use IsaacLab, because the repo's IsaacLab fork (0.30.x, Isaac 4.2-era) does not
load against Isaac Sim 5.1. Same self-contained pattern as g1_walk_render.py.

It loads the layout's usd_collection, spawns a Franka next to the pick target,
and loops a kinematic pick-and-place: hover over the pick object, descend,
"grasp" (the object follows the gripper), carry it over the place target and
release. No physics-based grasping — this is a task visualization, not training.

Env vars:
    SAGE_LAYOUT_DIR      results/<layout_id> directory (required)
    SAGE_LAYOUT_ID       layout id (required)
    SAGE_TASK_PICK       substring of the object id to pick   (default "apple")
    SAGE_TASK_PLACE      substring of the object id to place on (default "plate")
    SAGE_FRANKA_USD      local Franka USD override (else Isaac assets root)
    SAGE_RENDER_HEADLESS 1 = one cycle offscreen then exit (smoke test)
"""

import json
import math
import os
import sys

LAYOUT_DIR = os.environ.get("SAGE_LAYOUT_DIR", "")
LAYOUT_ID = os.environ.get("SAGE_LAYOUT_ID", "")
PICK_KEY = os.environ.get("SAGE_TASK_PICK", "apple").lower()
PLACE_KEY = os.environ.get("SAGE_TASK_PLACE", "plate").lower()
HEADLESS = os.environ.get("SAGE_RENDER_HEADLESS", "0") == "1"

if not LAYOUT_DIR or not os.path.isdir(LAYOUT_DIR):
    print(f"[franka_viz] SAGE_LAYOUT_DIR invalid: {LAYOUT_DIR}", file=sys.stderr)
    sys.exit(2)

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": HEADLESS, "enable_cameras": True})

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim


def log(msg):
    print(f"[franka_viz] {msg}", flush=True)


def euler_deg_to_quat_wxyz(rx, ry, rz):
    cr, sr = math.cos(math.radians(rx) / 2), math.sin(math.radians(rx) / 2)
    cp, sp = math.cos(math.radians(ry) / 2), math.sin(math.radians(ry) / 2)
    cy, sy = math.cos(math.radians(rz) / 2), math.sin(math.radians(rz) / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def load_transforms():
    path = os.path.join(LAYOUT_DIR, f"{LAYOUT_ID}_usd_collection", "rigid_object_transform_dict.json")
    with open(path) as f:
        return json.load(f)


def load_scene(stage, transforms):
    coll = os.path.join(LAYOUT_DIR, f"{LAYOUT_ID}_usd_collection")
    by_name = {}
    for f in sorted(os.listdir(coll)):
        name, ext = os.path.splitext(f)
        if ext in (".usd", ".usdz"):
            by_name.setdefault(name, {})[ext] = os.path.join(coll, f)
    for name, files in by_name.items():
        usd_path = files.get(".usd") or files.get(".usdz")
        prim_path = f"/World/scene/{name}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        # Freeze rigid objects: this is a kinematic visualization, gravity off.
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)
        if name.startswith(("floor_", "wall_", "door_", "window_")):
            continue
        tf = transforms.get(name)
        if tf is None:
            continue
        pos, rot = tf["position"], tf["rotation"]
        SingleXFormPrim(prim_path).set_world_pose(
            position=np.array([pos["x"], pos["y"], pos["z"]]),
            orientation=euler_deg_to_quat_wxyz(rot["x"], rot["y"], rot["z"]),
        )
    log(f"scene loaded: {len(by_name)} assets")


def object_top_z(layout, obj_id, base_z):
    for room in layout.get("rooms", []):
        for obj in room.get("objects", []):
            if obj.get("id") == obj_id:
                dims = obj.get("dimensions") or {}
                return base_z + float(dims.get("height", 0.08))
    return base_z + 0.08


def find_task_objects(transforms):
    pick_id = place_id = None
    for name in transforms:
        low = name.lower()
        if pick_id is None and PICK_KEY in low:
            pick_id = name
        if place_id is None and PLACE_KEY in low and PICK_KEY not in low:
            place_id = name
    return pick_id, place_id


def add_lights(stage, center, top):
    dome = UsdLux.DomeLight.Define(stage, "/World/SAGEDome")
    dome.CreateIntensityAttr(650.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/SAGEDistant")
    distant.CreateIntensityAttr(800.0)
    distant.CreateAngleAttr(0.5)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 35.0))
    rect = UsdLux.RectLight.Define(stage, "/World/SAGERect")
    rect.CreateIntensityAttr(900.0)
    rect.CreateWidthAttr(10.0)
    rect.CreateHeightAttr(8.0)
    xf = UsdGeom.Xformable(rect.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(top) + 1.5))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))


def hide_ceilings(stage):
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if "/World/scene/floor_" in path and path.endswith("_ceiling"):
            UsdGeom.Imageable(prim).MakeInvisible()


def resolve_franka_usd():
    override = os.environ.get("SAGE_FRANKA_USD")
    if override and os.path.exists(override):
        return override
    try:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
    except Exception:
        root = None
    if root:
        for rel in (
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "/Isaac/Robots/Franka/franka.usd",
        ):
            return root + rel  # first candidate; reference resolution will tell
    return None


def main():
    layout = json.load(open(os.path.join(LAYOUT_DIR, f"{LAYOUT_ID}.json")))
    transforms = load_transforms()

    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    load_scene(stage, transforms)
    hide_ceilings(stage)

    pick_id, place_id = find_task_objects(transforms)
    if not pick_id or not place_id:
        log(f"task objects not found (pick={pick_id}, place={place_id}) — showing scene only")
    pick_tf = transforms.get(pick_id) if pick_id else None
    place_tf = transforms.get(place_id) if place_id else None

    # Camera/lights framed on the task area (or scene center as fallback).
    if pick_tf:
        focus = np.array([pick_tf["position"]["x"], pick_tf["position"]["y"], pick_tf["position"]["z"]])
    else:
        focus = np.zeros(3)
    add_lights(stage, focus, focus[2] + 2.0)

    franka = None
    ik_solver = None
    if pick_tf and place_tf:
        franka_usd = resolve_franka_usd()
        log(f"franka usd: {franka_usd}")
        if franka_usd:
            # Base on the floor next to the pick object, facing it.
            room_center = focus * 0.0
            pick_xy = focus[:2]
            away = pick_xy - room_center[:2]
            away = away / (np.linalg.norm(away) + 1e-6)
            base_xy = pick_xy + away * 0.55
            yaw = math.degrees(math.atan2(*(pick_xy - base_xy)[::-1]))
            add_reference_to_stage(usd_path=franka_usd, prim_path="/World/franka")
            SingleXFormPrim("/World/franka").set_world_pose(
                position=np.array([base_xy[0], base_xy[1], 0.0]),
                orientation=euler_deg_to_quat_wxyz(0, 0, yaw),
            )
            franka = SingleArticulation("/World/franka", name="franka")
            world.scene.add(franka)

    world.reset()

    if franka is not None:
        try:
            from isaacsim.robot_motion.motion_generation import (
                ArticulationKinematicsSolver,
                LulaKinematicsSolver,
                interface_config_loader,
            )
            cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
            lula = LulaKinematicsSolver(**cfg)
            ik_solver = ArticulationKinematicsSolver(franka, lula, "right_gripper")
            log("Lula IK ready")
        except Exception as exc:
            log(f"IK unavailable ({exc}) — arm will stay at home pose")
            ik_solver = None

    cam_eye = focus + np.array([1.8, -1.8, 1.6])
    set_camera_view(eye=cam_eye.tolist(), target=focus.tolist())

    # Pick/place waypoints (world frame).
    seq = []
    pick_obj = None
    if pick_tf and place_tf and ik_solver is not None:
        p = np.array([pick_tf["position"]["x"], pick_tf["position"]["y"], pick_tf["position"]["z"]])
        q = np.array([place_tf["position"]["x"], place_tf["position"]["y"], place_tf["position"]["z"]])
        pick_top = object_top_z(layout, pick_id, p[2])
        place_top = object_top_z(layout, place_id, q[2])
        hover = 0.18
        seq = [
            ("move", np.array([p[0], p[1], pick_top + hover]), 90),
            ("move", np.array([p[0], p[1], pick_top + 0.015]), 60),
            ("grasp", None, 20),
            ("move", np.array([p[0], p[1], pick_top + hover]), 60),
            ("move", np.array([q[0], q[1], place_top + hover]), 110),
            ("move", np.array([q[0], q[1], place_top + 0.02]), 60),
            ("release", None, 20),
            ("move", np.array([q[0], q[1], place_top + hover]), 60),
        ]
        pick_obj = SingleXFormPrim(f"/World/scene/{pick_id}")
        pick_home, pick_home_rot = pick_obj.get_world_pose()
        log(f"task: {pick_id} -> {place_id}")

    def ee_pos():
        pos, _ = ik_solver.compute_end_effector_pose()
        return np.array(pos)

    attached = False
    cycle = 0
    while simulation_app.is_running():
        if not seq:
            world.step(render=True)
            continue
        if cycle > 0 and HEADLESS:
            break
        # reset object to start for each replay cycle
        if pick_obj is not None:
            pick_obj.set_world_pose(position=pick_home, orientation=pick_home_rot)
        attached = False
        current = ee_pos()
        for action, target, frames in seq:
            if not simulation_app.is_running():
                break
            if action == "grasp":
                attached = True
                continue
            if action == "release":
                attached = False
                continue
            start = ee_pos()
            for i in range(frames):
                alpha = (i + 1) / frames
                goal = start * (1 - alpha) + target * alpha
                ik_action, ok = ik_solver.compute_inverse_kinematics(
                    target_position=goal,
                    target_orientation=euler_deg_to_quat_wxyz(180, 0, 0),  # gripper down
                )
                if ok:
                    franka.apply_action(ik_action)
                if attached and pick_obj is not None:
                    grip = ee_pos()
                    pick_obj.set_world_pose(position=grip - np.array([0, 0, 0.02]))
                world.step(render=True)
        cycle += 1
        log(f"pick-and-place cycle {cycle} done")

    log("exiting")
    simulation_app.close()


if __name__ == "__main__":
    main()
