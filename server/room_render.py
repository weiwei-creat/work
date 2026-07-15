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
import os

from models import FloorPlan, Room, Wall
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np


_NVDIFFRAST_RENDERER = None
_NVDIFFRAST_IMPORT_ERROR = None
# CPU fallback for room rendering. Enabled by default because nvdiffrast/torch
# GPU rendering requires CUDA kernels for the host GPU's compute capability; on
# newer GPUs (e.g. RTX 5080 / sm_120) the pinned torch 2.5.1+cu124 has no kernel
# image and GPU rendering fails. The CPU path produces schematic top-down /
# perspective images the semantic critic can still reason over. Set
# SAGE_ENABLE_CPU_RENDER_FALLBACK=0 to force GPU-only (once torch supports the GPU).
_CPU_RENDER_FALLBACK_ENABLED = os.environ.get(
    "SAGE_ENABLE_CPU_RENDER_FALLBACK", "1"
).lower() in ("1", "true", "yes")


def _load_nvdiffrast_renderer():
    """Import nvdiffrast-backed helpers lazily so CPU fallback can still run."""
    global _NVDIFFRAST_RENDERER, _NVDIFFRAST_IMPORT_ERROR

    if _NVDIFFRAST_RENDERER is not None:
        return _NVDIFFRAST_RENDERER
    if _NVDIFFRAST_IMPORT_ERROR is not None:
        raise _NVDIFFRAST_IMPORT_ERROR

    try:
        from nvdiffrast_rendering.mesh import get_mesh_dict_list_from_single_room, build_mesh_dict
        from nvdiffrast_rendering.render import (
            rasterize_mesh_dict_list_with_uv_efficient,
            rasterize_mesh_dict_list_with_uv_efficient_uv_diff,
        )
        from nvdiffrast_rendering.camera import (
            get_camera_perspective_projection_matrix,
            get_intrinsic,
            build_camera_matrix,
            get_mvp_matrix,
            get_camera_orthogonal_projection_matrix,
            get_full_view_camera_sampling,
        )
        from nvdiffrast_rendering.context import get_glctx

        _NVDIFFRAST_RENDERER = {
            "get_mesh_dict_list_from_single_room": get_mesh_dict_list_from_single_room,
            "build_mesh_dict": build_mesh_dict,
            "rasterize_mesh_dict_list_with_uv_efficient": rasterize_mesh_dict_list_with_uv_efficient,
            "rasterize_mesh_dict_list_with_uv_efficient_uv_diff": rasterize_mesh_dict_list_with_uv_efficient_uv_diff,
            "get_camera_perspective_projection_matrix": get_camera_perspective_projection_matrix,
            "get_intrinsic": get_intrinsic,
            "build_camera_matrix": build_camera_matrix,
            "get_mvp_matrix": get_mvp_matrix,
            "get_camera_orthogonal_projection_matrix": get_camera_orthogonal_projection_matrix,
            "get_full_view_camera_sampling": get_full_view_camera_sampling,
            "get_glctx": get_glctx,
        }
        return _NVDIFFRAST_RENDERER
    except Exception as exc:
        _NVDIFFRAST_IMPORT_ERROR = exc
        raise


def _is_nvdiffrast_cuda_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "no kernel image is available for execution on the device" in message
        or "nvdiffrast" in message
        or "cuda error" in message
        or "no module named" in message
    )


def _raise_renderer_unavailable(exc: Exception):
    raise RuntimeError(
        "Isaac/RTX preview rendering failed and CPU fallback is disabled. "
        "Fix the Isaac Sim or nvdiffrast renderer instead of generating CPU fallback images."
    ) from exc


def _load_font(size: int, bold: bool = False):
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{font_name}", size)
    except Exception:
        return ImageFont.load_default()


def _object_color(obj):
    place_id = (obj.place_id or "").lower()
    if place_id == "wall" or place_id.startswith("wall"):
        return (99, 102, 241)
    if obj.place_id == "floor":
        return (37, 99, 235)
    return (20, 184, 166)


def _draw_cpu_top_down(
    layout: FloorPlan,
    room_id: str,
    resolution: int = 1024,
    draw_objects: bool = True,
    show_title: bool = True,
    margin_enabled: bool = True,
):
    room = next(room for room in layout.rooms if room.id == room_id)
    room_width = max(float(room.dimensions.width), 0.1)
    room_length = max(float(room.dimensions.length), 0.1)

    margin = max(36, int(resolution * 0.04)) if margin_enabled else 0
    scale = (resolution - 2 * margin) / max(room_width, room_length)
    image_width = max(256, int(room_width * scale + 2 * margin))
    image_height = max(256, int(room_length * scale + 2 * margin))

    img = Image.new("RGB", (image_width, image_height), (244, 241, 234))
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(13)
    small_font = _load_font(11)

    def to_px(x, y):
        rel_x = x - room.position.x
        rel_y = y - room.position.y
        return margin + rel_x * scale, image_height - margin - rel_y * scale

    floor_rect = [margin, margin, image_width - margin, image_height - margin]
    draw.rectangle(floor_rect, fill=(248, 247, 242, 255), outline=(40, 40, 40, 255), width=4)

    for x_m in range(0, int(room_width) + 1):
        x = margin + x_m * scale
        draw.line([(x, margin), (x, image_height - margin)], fill=(210, 210, 210, 160), width=1)
    for y_m in range(0, int(room_length) + 1):
        y = image_height - margin - y_m * scale
        draw.line([(margin, y), (image_width - margin, y)], fill=(210, 210, 210, 160), width=1)

    for wall in getattr(room, "walls", []):
        start = to_px(wall.start_point.x, wall.start_point.y)
        end = to_px(wall.end_point.x, wall.end_point.y)
        draw.line([start, end], fill=(28, 28, 28, 255), width=5)

    for opening, color in (
        (getattr(room, "doors", []), (234, 88, 12, 255)),
        (getattr(room, "windows", []), (14, 165, 233, 255)),
    ):
        for item in opening:
            pos = getattr(item, "position", None)
            dims = getattr(item, "dimensions", None)
            if pos is None or dims is None:
                continue
            cx, cy = to_px(pos.x, pos.y)
            w = max(8, float(getattr(dims, "width", 0.8)) * scale)
            h = max(8, float(getattr(dims, "length", 0.12)) * scale)
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=color)

    for obj in getattr(room, "objects", []):
        if not draw_objects:
            break
        if obj.place_id not in ("floor", "wall"):
            continue

        cx, cy = to_px(obj.position.x, obj.position.y)
        obj_w = max(8, float(obj.dimensions.width) * scale)
        obj_l = max(8, float(obj.dimensions.length) * scale)
        rotation = float(getattr(obj.rotation, "z", 0.0) or 0.0)
        if abs(rotation % 180 - 90) < 1:
            obj_w, obj_l = obj_l, obj_w

        x1, y1 = cx - obj_w / 2, cy - obj_l / 2
        x2, y2 = cx + obj_w / 2, cy + obj_l / 2
        fill = _object_color(obj) + (78,)
        outline = _object_color(obj) + (255,)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=3)

        if obj.place_id == "floor":
            rotation_rad = np.radians(rotation)
            arrow_len = min(42, max(14, min(obj_w, obj_l) * 0.75))
            end_x = cx + arrow_len * np.sin(rotation_rad)
            end_y = cy - arrow_len * np.cos(rotation_rad)
            draw.line([(cx, cy), (end_x, end_y)], fill=(220, 38, 38, 255), width=4)
            head = 7
            left = rotation_rad + np.radians(145)
            right = rotation_rad - np.radians(145)
            draw.polygon(
                [
                    (end_x, end_y),
                    (end_x + head * np.sin(left), end_y - head * np.cos(left)),
                    (end_x + head * np.sin(right), end_y - head * np.cos(right)),
                ],
                fill=(220, 38, 38, 255),
            )

        label = str(getattr(obj, "type", "object"))[:28]
        text_bbox = draw.textbbox((cx, cy), label, font=small_font, anchor="mm")
        draw.rectangle(
            [text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2],
            fill=(15, 23, 42, 210),
        )
        draw.text((cx, cy), label, fill=(255, 255, 255, 255), font=small_font, anchor="mm")

    if show_title:
        title = f"{room.room_type} | {room_width:.1f}m x {room_length:.1f}m | CPU top-down fallback"
        draw.text((max(margin, 8), 10), title, fill=(30, 41, 59, 255), font=font)

    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def get_camera_view_direction(camera_pos, lookat_pos):
    """Calculate normalized view direction from camera to lookat position."""
    direction = lookat_pos - camera_pos
    direction = direction / np.linalg.norm(direction)
    return direction


def get_wall_normal(wall: Wall, room: Room):
    """Calculate the outward normal vector of a wall."""
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    
    # Wall direction vector (along the wall)
    wall_dir = end - start
    wall_dir = wall_dir / np.linalg.norm(wall_dir)
    
    # Get perpendicular vector (normal to wall, pointing outward)
    # Assuming walls are vertical (z component is 0 for normal in xy plane)
    # Cross product with up vector to get outward normal
    up = np.array([0, 0, 1])
    normal = np.cross(wall_dir, up)
    
    # Determine if normal points inward or outward
    # Check if normal points away from room center
    room_center = np.array([
        room.position.x + room.dimensions.width / 2,
        room.position.y + room.dimensions.length / 2,
        room.position.z
    ])
    wall_center = (start + end) / 2
    to_center = room_center - wall_center
    
    # If normal points toward center, flip it
    if np.dot(normal[:2], to_center[:2]) > 0:
        normal = -normal
    
    return normal


def should_exclude_wall(wall: Wall, room: Room, camera_pos, lookat_pos):
    """Determine if a wall should be excluded based on camera view direction."""
    # Get wall normal (outward facing)
    wall_normal = get_wall_normal(wall, room)
    
    # Get camera view direction
    view_dir = get_camera_view_direction(camera_pos, lookat_pos)
    
    # If view direction and wall normal are opposing (dot product < 0),
    # the camera is looking at the wall from outside, so we should exclude it
    dot_product = np.dot(view_dir[:2], wall_normal[:2])
    
    # Exclude walls that the camera is facing (negative dot product means facing the wall)
    return dot_product < -0.3  # threshold to handle corner cases


def filter_mesh_info_dict_by_walls(mesh_info_dict, walls_to_exclude):
    """Remove specific walls from the mesh_info_dict."""
    filtered_dict = {}
    excluded_wall_ids = {wall.id for wall in walls_to_exclude}
    
    for mesh_id, mesh_info in mesh_info_dict.items():
        # Check if this is a wall mesh and if it should be excluded
        is_excluded_wall = any(wall_id in mesh_id for wall_id in excluded_wall_ids)
        
        if not is_excluded_wall:
            filtered_dict[mesh_id] = mesh_info
    
    return filtered_dict


def get_filtered_mesh_dict_list(layout: FloorPlan, room: Room, camera_pos, lookat_pos):
    renderer = _load_nvdiffrast_renderer()
    build_mesh_dict = renderer["build_mesh_dict"]
    from tex_utils import export_single_room_layout_to_mesh_dict_list

    """Get mesh_dict_list with front walls removed based on camera angle."""
    # Get the original mesh_info_dict
    mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, room.id)
    
    # Determine which walls to exclude
    walls_to_exclude = [
        wall for wall in room.walls 
        if should_exclude_wall(wall, room, camera_pos, lookat_pos)
    ]
    
    # Filter the mesh_info_dict
    filtered_mesh_info_dict = filter_mesh_info_dict_by_walls(mesh_info_dict, walls_to_exclude)
    
    # Convert to mesh_dict_list
    mesh_dict_list = []
    for mesh_info in filtered_mesh_info_dict.values():
        vertices = mesh_info["mesh"].vertices
        faces = mesh_info["mesh"].faces
        vts = mesh_info["texture"]["vts"]
        fts = mesh_info["texture"]["fts"]
        texture_map_pil = Image.open(mesh_info["texture"]["texture_map_path"])
        # H_tex, W_tex = texture_map_pil.height, texture_map_pil.width
        # # TODO: resize to nearest power of 2 dimensions
        # def next_power_of_2(x):
        #     return 2 ** (int(x - 1).bit_length())
        
        # H_tex_pow2 = next_power_of_2(H_tex)
        # W_tex_pow2 = next_power_of_2(W_tex)
        
        # # Resize texture to power-of-2 if necessary
        # if H_tex != H_tex_pow2 or W_tex != W_tex_pow2:
        #     texture_map_pil = texture_map_pil.resize((W_tex_pow2, H_tex_pow2), Image.LANCZOS)
        texture_map = np.array(texture_map_pil) / 255.0
        
        mesh_dict = build_mesh_dict(vertices, faces, vts, fts, texture_map)
        mesh_dict_list.append(mesh_dict)
    
    return mesh_dict_list


def render_room_four_top_view(layout: FloorPlan, room_id: str, resolution = 768):
    try:
        import torch

        renderer = _load_nvdiffrast_renderer()
        get_mesh_dict_list_from_single_room = renderer["get_mesh_dict_list_from_single_room"]
        get_intrinsic = renderer["get_intrinsic"]
        get_camera_perspective_projection_matrix = renderer["get_camera_perspective_projection_matrix"]
        get_glctx = renderer["get_glctx"]
        build_camera_matrix = renderer["build_camera_matrix"]
        get_mvp_matrix = renderer["get_mvp_matrix"]
        rasterize_mesh_dict_list_with_uv_efficient = renderer["rasterize_mesh_dict_list_with_uv_efficient"]
    except Exception as exc:
        if _is_nvdiffrast_cuda_failure(exc):
            if not _CPU_RENDER_FALLBACK_ENABLED:
                _raise_renderer_unavailable(exc)
            print(f"Warning: nvdiffrast unavailable, using CPU top-view fallback: {exc}", flush=True)
            base = _draw_cpu_top_down(layout, room_id, resolution=resolution)
            return [np.rot90(base, k).copy() for k in range(4)]
        raise

    try:
        mesh_dict_list = get_mesh_dict_list_from_single_room(layout, room_id)
        intrinsic = get_intrinsic(80, resolution, resolution)
        projection_matrix = get_camera_perspective_projection_matrix(
            intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], resolution, resolution, 0.001, 100.0)

        glctx = get_glctx()

        all_rooms = layout.rooms
        room = next(room for room in all_rooms if room.id == room_id)

        room_position = np.array([room.position.x, room.position.y, room.position.z])
        room_height = room.dimensions.height * 1.5
        room_width = room.dimensions.width
        room_length = room.dimensions.length

        room_top_corners = [
            room_position + np.array([0, 0, room_height]),
            room_position + np.array([room_width, 0, room_height]),
            room_position + np.array([0, room_length, room_height]),
            room_position + np.array([room_width, room_length, room_height]),
        ]

        room_lookat_corners = [
            room_position + np.array([room_width * 0.5, room_length * 0.5, 0]),
            room_position + np.array([room_width * 0.5, room_length * 0.5, 0]),
            room_position + np.array([room_width * 0.5, room_length * 0.5, 0]),
            room_position + np.array([room_width * 0.5, room_length * 0.5, 0]),
        ]

        all_rgb = []
        for top_corner, lookat_corner in zip(room_top_corners, room_lookat_corners):
            camera_matrix = build_camera_matrix(
                torch.from_numpy(top_corner).float(),
                torch.from_numpy(lookat_corner).float(),
                torch.from_numpy(np.array([0, 0, 1])).float()
            )

            mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)
            valid, instance_id, rgb = rasterize_mesh_dict_list_with_uv_efficient(
                mesh_dict_list, mvp_matrix, glctx, (resolution, resolution)
            )
            rgb = rgb.cpu().numpy().clip(0, 1)
            all_rgb.append(rgb)

        return all_rgb
    except Exception as exc:
        if _is_nvdiffrast_cuda_failure(exc):
            if not _CPU_RENDER_FALLBACK_ENABLED:
                _raise_renderer_unavailable(exc)
            print(f"Warning: nvdiffrast top-view render failed, using CPU fallback: {exc}", flush=True)
            base = _draw_cpu_top_down(layout, room_id, resolution=resolution)
            return [np.rot90(base, k).copy() for k in range(4)]
        raise


def render_room_four_edges_view(layout: FloorPlan, room_id: str, resolution = 1024):
    try:
        import torch
        renderer = _load_nvdiffrast_renderer()
        get_intrinsic = renderer["get_intrinsic"]
        get_camera_perspective_projection_matrix = renderer["get_camera_perspective_projection_matrix"]
        get_glctx = renderer["get_glctx"]
        get_full_view_camera_sampling = renderer["get_full_view_camera_sampling"]
        build_camera_matrix = renderer["build_camera_matrix"]
        get_mvp_matrix = renderer["get_mvp_matrix"]
        rasterize_mesh_dict_list_with_uv_efficient = renderer["rasterize_mesh_dict_list_with_uv_efficient"]
    except Exception as exc:
        if _is_nvdiffrast_cuda_failure(exc):
            if not _CPU_RENDER_FALLBACK_ENABLED:
                _raise_renderer_unavailable(exc)
            print(f"Warning: nvdiffrast unavailable, using CPU room render fallback: {exc}", flush=True)
            base = _draw_cpu_top_down(layout, room_id, resolution=resolution)
            return [np.rot90(base, k).copy() for k in range(4)]
        raise
    
    try:
        fov = 35.0
        aspect_ratio = 16 / 9
        res_width = resolution
        res_height = int(res_width / aspect_ratio)

        all_rooms = layout.rooms
        room = next(room for room in all_rooms if room.id == room_id)

        room_position = np.array([room.position.x, room.position.y, room.position.z])
        room_height = room.dimensions.height
        room_width = room.dimensions.width
        room_length = room.dimensions.length
        room_center = np.array([room_position[0] + room_width/2, room_position[1] + room_length/2, room_height/2]).reshape(-1).tolist()
        room_scales = np.array([room_width, room_length, room_height]).reshape(-1).tolist()


        intrinsic = get_intrinsic(fov, res_height, res_width)
        projection_matrix = get_camera_perspective_projection_matrix(
            intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3],
            res_height, res_width, 0.01, np.array(room_scales).max() * 5.0
        )

        glctx = get_glctx()


        horizontal_angle_list = [0, 90, 180, 270]
        vertical_angle = 45.0


        # for every edge of the room, build a camera pose
        all_rgb = []
        for horizontal_angle in horizontal_angle_list:
            camera_pos, camera_lookat, _ = get_full_view_camera_sampling(
                room_center, room_scales, (res_height, res_width),
                horizontal_angle, vertical_angle, fov, mode="adjustable"
            )

            camera_matrix = build_camera_matrix(
                torch.from_numpy(camera_pos).float(),
                torch.from_numpy(camera_lookat).float(),
                torch.from_numpy(np.array([0, 0, 1])).float()
            )

            mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)

            # Get filtered mesh_dict_list with front wall removed based on camera angle
            mesh_dict_list = get_filtered_mesh_dict_list(layout, room, camera_pos, camera_lookat)

            valid, instance_id, rgb = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, mvp_matrix, glctx, (res_height, res_width))
            rgb = rgb.cpu().numpy().clip(0, 1)

            all_rgb.append(rgb)

        return all_rgb
    except Exception as exc:
        if _is_nvdiffrast_cuda_failure(exc):
            if not _CPU_RENDER_FALLBACK_ENABLED:
                _raise_renderer_unavailable(exc)
            print(f"Warning: nvdiffrast render failed, using CPU room render fallback: {exc}", flush=True)
            base = _draw_cpu_top_down(layout, room_id, resolution=resolution)
            return [np.rot90(base, k).copy() for k in range(4)]
        raise




def render_room_top_orthogonal_view(layout: FloorPlan, room_id: str, resolution = 1024):
    try:
        import torch
        renderer = _load_nvdiffrast_renderer()
        get_mesh_dict_list_from_single_room = renderer["get_mesh_dict_list_from_single_room"]
        get_glctx = renderer["get_glctx"]
        build_camera_matrix = renderer["build_camera_matrix"]
        get_camera_orthogonal_projection_matrix = renderer["get_camera_orthogonal_projection_matrix"]
        get_mvp_matrix = renderer["get_mvp_matrix"]
        rasterize_mesh_dict_list_with_uv_efficient = renderer["rasterize_mesh_dict_list_with_uv_efficient"]
    except Exception as exc:
        if _is_nvdiffrast_cuda_failure(exc):
            if not _CPU_RENDER_FALLBACK_ENABLED:
                _raise_renderer_unavailable(exc)
            print(f"Warning: nvdiffrast unavailable, using CPU top-down fallback: {exc}", flush=True)
            return _draw_cpu_top_down(
                layout,
                room_id,
                resolution=resolution,
                draw_objects=False,
                show_title=False,
                margin_enabled=False,
            )
        raise

    try:
        mesh_dict_list = get_mesh_dict_list_from_single_room(layout, room_id)

        glctx = get_glctx()

        all_rooms = layout.rooms
        room = next(room for room in all_rooms if room.id == room_id)

        room_position = np.array([room.position.x, room.position.y, room.position.z])
        room_height = room.dimensions.height * 1.5
        room_width = room.dimensions.width
        room_length = room.dimensions.length

        camera_position = np.array([room_position[0] + room_width/2, room_position[1] + room_length/2, room_height])
        camera_lookat = np.array([room_position[0] + room_width/2, room_position[1] + room_length/2, 0])
        camera_up = np.array([0, 1, 0])

        camera_matrix = build_camera_matrix(
            torch.from_numpy(camera_position).float(),
            torch.from_numpy(camera_lookat).float(),
            torch.from_numpy(camera_up).float()
        )

        projection_matrix = get_camera_orthogonal_projection_matrix(0.001, 100.0, room_width * 0.5, room_length * 0.5)

        mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)

        # H, W should be the same ratio as the room width and length and should be no larger than 2048
        W = int(room_width * 1024)
        H = int(room_length * 1024)

        if max(H, W) > resolution:
            scale = resolution / max(H, W)
            H = int(H * scale)
            W = int(W * scale)

        valid, instance_id, rgb = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, mvp_matrix, glctx, (H, W))

        rgb = rgb.cpu().numpy().clip(0, 1)

        return rgb
    except Exception as exc:
        if _is_nvdiffrast_cuda_failure(exc):
            if not _CPU_RENDER_FALLBACK_ENABLED:
                _raise_renderer_unavailable(exc)
            print(f"Warning: nvdiffrast top-down render failed, using CPU fallback: {exc}", flush=True)
            return _draw_cpu_top_down(
                layout,
                room_id,
                resolution=resolution,
                draw_objects=False,
                show_title=False,
                margin_enabled=False,
            )
        raise
