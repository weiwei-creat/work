"""Open a generated SAGE scene USD in an Isaac Sim GUI window for inspection.

This is a self-contained SimulationApp script (run it with Isaac's python.sh, NOT
the conda env) so it boots its own renderer. It must not run at the same time as
another GUI Isaac instance on a 16GB GPU — the web console stops the headless MCP
kit before launching this viewer.

Usage:
    python.sh scene_viewer.py <usd_path>

The viewer guarantees the scene is lit: if the stage has no light prims (the raw
exported scene USDs contain only geometry), it adds a dome + distant + overhead
rect light so the GUI display isn't pitch black.
"""

import os
import sys

if len(sys.argv) < 2:
    print("Usage: scene_viewer.py <usd_path>", file=sys.stderr)
    sys.exit(2)

USD_PATH = os.path.abspath(sys.argv[1])
if not os.path.isfile(USD_PATH):
    print(f"[scene_viewer] USD not found: {USD_PATH}", file=sys.stderr)
    sys.exit(2)

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux


def ensure_lights(stage):
    """Add preview lights if the stage has none (raw scene USDs are unlit)."""
    has_light = any(
        prim.HasAPI(UsdLux.LightAPI) if hasattr(UsdLux, "LightAPI") else prim.IsA(UsdLux.DomeLight)
        for prim in stage.Traverse()
        if prim.GetTypeName().endswith("Light")
    )
    # Simpler, version-tolerant check: any prim whose type name contains "Light".
    has_light = any("Light" in prim.GetTypeName() for prim in stage.Traverse())
    if has_light:
        print("[scene_viewer] stage already has lights")
        return

    print("[scene_viewer] no lights found — adding viewer lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/SAGEViewerDomeLight")
    dome.CreateIntensityAttr(650.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/SAGEViewerDistantLight")
    distant.CreateIntensityAttr(800.0)
    distant.CreateAngleAttr(0.5)
    UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 35.0))

    # Overhead fill above the scene bounds.
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bound = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World"))
        box = bound.ComputeAlignedRange()
        cx = (box.GetMin()[0] + box.GetMax()[0]) / 2.0
        cy = (box.GetMin()[1] + box.GetMax()[1]) / 2.0
        top = box.GetMax()[2]
    except Exception:
        cx, cy, top = 0.0, 0.0, 3.0
    rect = UsdLux.RectLight.Define(stage, "/World/SAGEViewerRectLight")
    rect.CreateIntensityAttr(900.0)
    rect.CreateWidthAttr(12.0)
    rect.CreateHeightAttr(10.0)
    xf = UsdGeom.Xformable(rect.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, float(top) + 1.5))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))


print(f"[scene_viewer] opening {USD_PATH}")
ctx = omni.usd.get_context()
ctx.open_stage(USD_PATH)
for _ in range(5):
    simulation_app.update()

stage = ctx.get_stage()
if stage is not None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    ensure_lights(stage)

print("[scene_viewer] ready — close the Isaac Sim window to exit")
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
