"""
Convert USD mesh files to OBJ format using the usd-core library.

Usage:
    python usd_to_obj.py --usd_path /path/to/Aligned.usd --obj_path /path/to/output.obj

This script requires the `usd-core` package and should be run in an environment
that has it installed (e.g. genie-scene-generator conda env).
"""

import argparse
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom


def _triangulate_corners(face_vertex_counts, face_vertex_indices, uvs):
    """
    Triangulate polygon faces and produce per-corner (vertex_idx, uv) tuples.

    Returns lists of equal length: vertex_indices, uv_coords.
    """
    tri_verts = []
    tri_uvs = []
    corner = 0
    for count in face_vertex_counts:
        if count < 3:
            corner += count
            continue
        v0 = face_vertex_indices[corner]
        uv0 = uvs[corner] if uvs is not None else (0, 0)
        for j in range(1, count - 1):
            tri_verts.append(v0)
            tri_uvs.append(uv0)

            v1 = face_vertex_indices[corner + j]
            tri_verts.append(v1)
            tri_uvs.append(uvs[corner + j] if uvs is not None else (0, 0))

            v2 = face_vertex_indices[corner + j + 1]
            tri_verts.append(v2)
            tri_uvs.append(uvs[corner + j + 1] if uvs is not None else (0, 0))
        corner += count
    return tri_verts, tri_uvs


def usd_to_obj(usd_path, obj_path):
    """
    Convert a USD mesh file to OBJ format.

    Returns a dict with mesh statistics, or None on failure.
    """
    if not os.path.exists(usd_path):
        print(f"ERROR: USD file not found: {usd_path}", file=sys.stderr)
        return None

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"ERROR: Failed to open USD stage: {usd_path}", file=sys.stderr)
        return None

    all_points = []
    all_tri_verts = []
    all_tri_uvs = []
    mesh_count = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not points or not face_counts or face_indices is None:
            continue

        points = np.array(points, dtype=np.float64)
        f_counts = np.array(face_counts, dtype=np.int64)
        f_indices = np.array(face_indices, dtype=np.int64)

        # Get faceVarying UVs
        uv_array = None
        st_attr = mesh.GetPrim().GetAttribute("primvars:st")
        if st_attr:
            uv_data = st_attr.Get()
            if uv_data is not None and len(uv_data) > 0:
                uv_array = np.array(uv_data, dtype=np.float64)[:, :2]

        tri_verts, tri_uvs = _triangulate_corners(f_counts, f_indices, uv_array)
        if len(tri_verts) == 0:
            continue

        all_points.append(points)
        all_tri_verts.extend(tri_verts)
        all_tri_uvs.extend(tri_uvs)
        mesh_count += 1

    if mesh_count == 0:
        print(f"ERROR: No mesh prims found in {usd_path}", file=sys.stderr)
        return None

    # Concatenate all point arrays
    merged_points = np.vstack(all_points)

    # Build per-triangle-corner vertex data (flattened: 3 verts per triangle)
    tri_verts = np.array(all_tri_verts, dtype=np.int64)
    tri_uvs = np.array(all_tri_uvs, dtype=np.float64)
    num_corners = len(tri_verts)

    # Write OBJ file
    os.makedirs(os.path.dirname(obj_path) if os.path.dirname(obj_path) else ".", exist_ok=True)
    with open(obj_path, "w") as f:
        f.write(f"# Generated from: {usd_path}\n")
        f.write(f"# Vertices: {len(merged_points)}, Triangles: {num_corners // 3}\n")

        for v in merged_points:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for uv in tri_uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        # Each triangle corner has its own v and vt entry (1-indexed)
        for i in range(0, num_corners, 3):
            vi, vj, vk = tri_verts[i] + 1, tri_verts[i + 1] + 1, tri_verts[i + 2] + 1
            ti, tj, tk = i + 1, i + 2, i + 3
            f.write(f"f {vi}/{ti} {vj}/{tj} {vk}/{tk}\n")

    return {
        "num_vertices": int(len(merged_points)),
        "num_faces": num_corners // 3,
        "num_meshes": mesh_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert USD mesh to OBJ")
    parser.add_argument("--usd_path", required=True, help="Path to the USD file")
    parser.add_argument("--obj_path", required=True, help="Path for the output OBJ file")
    args = parser.parse_args()

    result = usd_to_obj(args.usd_path, args.obj_path)
    if result is None:
        sys.exit(1)
    print(f"Converted: {result['num_vertices']} verts, {result['num_faces']} faces, {result['num_meshes']} meshes")


if __name__ == "__main__":
    main()
