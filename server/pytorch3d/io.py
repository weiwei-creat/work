"""Minimal OBJ writer compatible with the PyTorch3D ``save_obj`` call used here."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def save_obj(f, verts, faces, verts_uvs=None, faces_uvs=None, texture_map=None, **_kwargs):
    """Write a Wavefront OBJ file for the subset needed by SAGE.

    PyTorch3D is heavy to install on newer CUDA stacks, while this project only
    imports ``save_obj``. The implementation keeps the same 0-based tensor input
    convention and writes 1-based OBJ indices.
    """
    path = Path(f)
    path.parent.mkdir(parents=True, exist_ok=True)

    verts_np = _to_numpy(verts).reshape(-1, 3)
    faces_np = _to_numpy(faces).reshape(-1, 3).astype(np.int64)
    verts_uvs_np = _to_numpy(verts_uvs)
    faces_uvs_np = _to_numpy(faces_uvs)

    has_uvs = verts_uvs_np is not None and faces_uvs_np is not None
    if has_uvs:
        verts_uvs_np = verts_uvs_np.reshape(-1, 2)
        faces_uvs_np = faces_uvs_np.reshape(-1, 3).astype(np.int64)

    with path.open("w", encoding="utf-8") as obj:
        if texture_map is not None:
            obj.write(f"# texture_map provided by caller; see adjacent texture files\n")

        for vertex in verts_np:
            obj.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")

        if has_uvs:
            for uv in verts_uvs_np:
                obj.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")

        for index, face in enumerate(faces_np):
            if has_uvs:
                uv_face = faces_uvs_np[index]
                entries = [
                    f"{int(v_idx) + 1}/{int(uv_idx) + 1}"
                    for v_idx, uv_idx in zip(face, uv_face)
                ]
            else:
                entries = [str(int(v_idx) + 1) for v_idx in face]
            obj.write(f"f {' '.join(entries)}\n")
