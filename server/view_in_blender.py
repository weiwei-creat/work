"""
Interactive Blender viewer for SAGE results.
Self-contained — no project imports needed, reads directly from results/ directory.

Usage:
    cd /home/gaok/coding/sage/server
    blender --python view_in_blender.py -- --layout-id layout_1aaf70ef

Or in Blender Scripting workspace, edit LAYOUT_ID below and click Run Script.
"""

import bpy
import mathutils
import json
import os
import sys
import argparse
# ── Configuration ────────────────────────────────────────────
LAYOUT_ID = "layout_1aaf70ef"   # <-- change to your layout
ROOM_ID = None                   # specific room, or None for all
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ── Blender helpers ──────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def get_or_create_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def make_material(name, texture_path, fallback_color=(0.7, 0.7, 0.7, 1.0)):
    """Create a Principled BSDF material, textured if possible, else solid color."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(output.inputs['Surface'], bsdf.outputs['BSDF'])

    if texture_path and os.path.exists(texture_path):
        tex = nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(texture_path)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex.outputs['Color'])
    else:
        bsdf.inputs['Base Color'].default_value = fallback_color
    return mat


def mesh_from_data(name, vertices, faces, collection):
    me = bpy.data.meshes.new(name=name)
    me.from_pydata(vertices, [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    return obj


# ── Room structure ───────────────────────────────────────────

def create_room_structure(room, layout_id, col):
    """Build walls + floor from room geometry in the layout JSON."""
    layout_dir = os.path.join(RESULTS_DIR, layout_id)
    materials_dir = os.path.join(layout_dir, "materials")

    # Find material textures
    wall_mat = None
    floor_mat = None
    if os.path.isdir(materials_dir):
        for fname in sorted(os.listdir(materials_dir)):
            if fname.endswith("_wall.png") and wall_mat is None:
                wall_mat = make_material("wall_material", os.path.join(materials_dir, fname), (0.85, 0.83, 0.78, 1.0))
            if fname.endswith("_floor.png") and floor_mat is None:
                floor_mat = make_material("floor_material", os.path.join(materials_dir, fname), (0.65, 0.55, 0.45, 1.0))
    if wall_mat is None:
        wall_mat = make_material("wall_material", None, (0.85, 0.83, 0.78, 1.0))
    if floor_mat is None:
        floor_mat = make_material("floor_material", None, (0.65, 0.55, 0.45, 1.0))

    walls = room.get("walls", [])
    pos = room.get("position", {})
    dims = room.get("dimensions", {})
    rw, rl, rh = dims.get("width", 4), dims.get("length", 4), dims.get("height", 3)
    px, py, pz = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)

    if walls:
        for wall in walls:
            sp, ep = wall["start_point"], wall["end_point"]
            sx, sy = sp["x"], sp["y"]
            ex, ey = ep["x"], ep["y"]
            thick = wall.get("thickness", 0.1)
            h = wall.get("height", rh)

            dx, dy = ex - sx, ey - sy
            length = (dx**2 + dy**2) ** 0.5
            if length < 0.001:
                continue
            nx, ny = -dy / length, dx / length  # outward perpendicular

            hw = thick / 2
            v = [
                [sx - nx * hw, sy - ny * hw, 0],
                [sx + nx * hw, sy + ny * hw, 0],
                [ex + nx * hw, ey + ny * hw, 0],
                [ex - nx * hw, ey - ny * hw, 0],
                [sx - nx * hw, sy - ny * hw, h],
                [sx + nx * hw, sy + ny * hw, h],
                [ex + nx * hw, ey + ny * hw, h],
                [ex - nx * hw, ey - ny * hw, h],
            ]
            f = [
                [0, 1, 2], [0, 2, 3],
                [4, 7, 6], [4, 6, 5],
                [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3],
                [3, 7, 4], [3, 4, 0],
            ]
            obj = mesh_from_data(f"wall_{wall.get('id', '?')}", v, f, col)
            obj.data.materials.append(wall_mat)

    # Floor
    fv = [
        [px, py, pz], [px, py + rl, pz],
        [px + rw, py + rl, pz], [px + rw, py, pz],
    ]
    ff = [[0, 2, 1], [0, 3, 2]]
    floor = mesh_from_data(f"floor_{room.get('id', '?')}", fv, ff, col)
    floor.data.materials.append(floor_mat)


# ── Doors ────────────────────────────────────────────────────

def create_door_mesh(door, wall, layout_id, col):
    """Create a door box at the right position on a wall."""
    sp, ep = wall["start_point"], wall["end_point"]
    sx, sy = sp["x"], sp["y"]
    ex, ey = ep["x"], ep["y"]

    dx, dy = ex - sx, ey - sy
    wlen = (dx**2 + dy**2) ** 0.5
    if wlen < 0.001:
        return
    ux, uy = dx / wlen, dy / wlen
    nx, ny = -uy, ux

    pos_on_wall = door.get("position_on_wall", 0.5)
    dw = door.get("width", 0.9)
    dh = door.get("height", 2.0)
    thick = wall.get("thickness", 0.1)

    cx = sx + dx * pos_on_wall
    cy = sy + dy * pos_on_wall
    hw = dw / 2
    ht = thick * 0.4  # door thickness

    v = [
        [cx - ux * hw - nx * ht, cy - uy * hw - ny * ht, 0.0],
        [cx + ux * hw - nx * ht, cy + uy * hw - ny * ht, 0.0],
        [cx + ux * hw + nx * ht, cy + uy * hw + ny * ht, 0.0],
        [cx - ux * hw + nx * ht, cy - uy * hw + ny * ht, 0.0],
        [cx - ux * hw - nx * ht, cy - uy * hw - ny * ht, dh],
        [cx + ux * hw - nx * ht, cy + uy * hw - ny * ht, dh],
        [cx + ux * hw + nx * ht, cy + uy * hw + ny * ht, dh],
        [cx - ux * hw + nx * ht, cy - uy * hw + ny * ht, dh],
    ]
    f = [[0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
         [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
         [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]

    obj = mesh_from_data(f"door_{door.get('id', '?')}", v, f, col)

    door_mat_name = door.get("door_material", "")
    tex_path = os.path.join(RESULTS_DIR, layout_id, "materials", f"{door_mat_name}_texture.png")
    if door_mat_name and os.path.exists(tex_path):
        mat = make_material(f"door_mat_{door_mat_name}", tex_path)
    else:
        mat = make_material("door_default", None, (0.4, 0.25, 0.15, 1.0))
    obj.data.materials.append(mat)


# ── Objects from OBJ files ───────────────────────────────────

def load_ply(path):
    """Load a binary PLY file, return (vertices, faces)."""
    import struct

    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8', errors='replace').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

    n_verts = n_faces = 0
    for line in header_lines:
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])

    # Parse vertex and face properties
    vprops = []  # list of (type, name) for vertex section
    fps = []     # list of (type, name) for face section
    current_section = 'vertex'
    for line in header_lines:
        if line.startswith('element face'):
            current_section = 'face'
            continue
        if line.startswith('property '):
            parts = line.split()
            if current_section == 'vertex':
                vprops.append((parts[1], parts[2]))
            elif current_section == 'face' and parts[1] == 'list':
                fps.append(('list', parts[2], parts[3], parts[4]))

    # Compute vertex stride and field offsets
    vsize = 0
    xoff = yoff = zoff = -1
    for ptype, pname in vprops:
        size = 4 if ptype in ('float', 'int', 'uint') else (8 if ptype == 'double' else 1)
        if pname == 'x': xoff = vsize
        elif pname == 'y': yoff = vsize
        elif pname == 'z': zoff = vsize
        vsize += size

    # Read binary body
    header_bytes = '\n'.join(header_lines) + '\n'
    with open(path, 'rb') as f:
        data = f.read()
    body = data[len(header_bytes.encode()):]

    # Read vertices
    verts = []
    for vi in range(n_verts):
        row = vi * vsize
        x = struct.unpack_from('f', body, row + xoff)[0]
        y = struct.unpack_from('f', body, row + yoff)[0]
        z = struct.unpack_from('f', body, row + zoff)[0]
        verts.append([x, y, z])

    # Read faces
    fpos = n_verts * vsize
    faces = []
    for _ in range(n_faces):
        count = body[fpos]
        fpos += 1
        face = []
        for _ in range(count):
            vi = struct.unpack_from('I', body, fpos)[0]
            fpos += 4
            face.append(vi)
        faces.append(face)

    return verts, faces


def load_room_objects(room, layout_id, col):
    """Load objects from .ply files with textures + UVs from pickle files."""
    import pickle

    layout_dir = os.path.join(RESULTS_DIR, layout_id)
    loaded = 0

    for obj_data in room.get("objects", []):
        source_id = obj_data.get("source_id", "")
        source = obj_data.get("source", "objaverse")
        obj_id = obj_data.get("id", source_id)
        pos = obj_data.get("position", {})
        euler = obj_data.get("euler", {})

        ply_path = os.path.join(layout_dir, source, f"{source_id}.ply")
        if not os.path.exists(ply_path):
            continue

        try:
            verts, faces = load_ply(ply_path)
            if not verts:
                continue
        except Exception as e:
            print(f"  PLY load failed for {source_id}: {e}")
            continue

        # Create mesh from PLY data
        obj = mesh_from_data(f"obj_{obj_id}", verts, faces, col)
        obj.location = (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
        obj.rotation_euler = (euler.get("x", 0), euler.get("y", 0), euler.get("z", 0))

        # Load texture and UVs
        tex_path = os.path.join(layout_dir, source, f"{source_id}_texture.png")
        coords_path = os.path.join(layout_dir, source, f"{source_id}_tex_coords.pkl")

        mat = make_material(f"mat_{source_id}",
                            tex_path if os.path.exists(tex_path) else None)

        # Load UV coordinates from pickle
        if os.path.exists(coords_path):
            try:
                with open(coords_path, 'rb') as f:
                    tc = pickle.load(f)
                vts_data = tc.get("vts")
                fts_data = tc.get("fts")
                if vts_data is not None and fts_data is not None:
                    uv_layer = obj.data.uv_layers.new(name="UVMap")
                    loop_idx = 0
                    for face_vtx_indices in fts_data:
                        for vi in face_vtx_indices:
                            if loop_idx < len(uv_layer.data):
                                uv_layer.data[loop_idx].uv = (
                                    float(vts_data[vi][0]),
                                    1.0 - float(vts_data[vi][1]),
                                )
                            loop_idx += 1
            except Exception:
                pass

        obj.data.materials.append(mat)
        loaded += 1

    return loaded


# ── Main import ──────────────────────────────────────────────

def import_layout(layout_id, room_id=None):
    json_path = os.path.join(RESULTS_DIR, layout_id, f"{layout_id}.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        return

    with open(json_path) as f:
        layout = json.load(f)
    print(f"Layout: {layout_id} ({len(layout.get('rooms', []))} rooms)")

    clear_scene()

    # --- Lighting ---
    sun = bpy.data.lights.new("Sun", 'SUN')
    sun.energy = 8.0
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.location = (20, 20, 25)
    sun_obj.rotation_euler = (0.8, 0.3, 0.8)

    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.85, 0.88, 0.95, 1.0)
    bg.inputs['Strength'].default_value = 0.5

    # --- Load each room ---
    total_objects = 0
    for room in layout.get("rooms", []):
        rid = room.get("id", "?")
        if room_id and rid != room_id:
            continue

        rtype = room.get("room_type", "unknown")
        col = get_or_create_collection(f"room_{rid}_{rtype}")

        # Walls + floor
        create_room_structure(room, layout_id, col)

        # Doors
        wall_by_id = {w["id"]: w for w in room.get("walls", [])}
        for door in room.get("doors", []):
            wall = wall_by_id.get(door.get("wall_id", ""))
            if wall:
                create_door_mesh(door, wall, layout_id, col)

        # Placed objects
        n = load_room_objects(room, layout_id, col)
        total_objects += n
        print(f"  Room {rid}: {n} objects")

    print(f"Total objects: {total_objects}")

    # --- Viewport setup ---
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    if bpy.context.view_layer.objects:
        bpy.ops.object.select_all(action='SELECT')
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with bpy.context.temp_override(area=area, region=region):
                            bpy.ops.view3d.view_selected()
                        break
                break

    print("Ready.")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-id", type=str, default=LAYOUT_ID)
    parser.add_argument("--room-id", type=str, default=ROOM_ID)
    args = parser.parse_args(argv)

    import_layout(args.layout_id, args.room_id)
