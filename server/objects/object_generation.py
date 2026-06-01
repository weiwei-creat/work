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
ssh -L 14412:gpu-h100-0272:14412 hongchix@pdx
"""

import requests
import json
import os
import sys
import time
from pathlib import Path
import sys
import numpy as np
import trimesh
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR

try:
    from key import SERVER_URL
    from objects.load_glb import load_glb_advanced
    from objects.object_attribute_inference import (
        infer_attributes_from_claude,
        estimate_front_from_mesh
    )
except ImportError:
    sys.path.insert(0, SERVER_ROOT_DIR)
    from key import SERVER_URL
    from objects.load_glb import load_glb_advanced
    from objects.object_attribute_inference import (
        infer_attributes_from_claude,
        estimate_front_from_mesh
    )

import torch
from pytorch3d.io import save_obj


def _skip_object_render_analysis():
    return os.environ.get("SAGE_SKIP_OBJECT_RENDER_ANALYSIS", "0").lower() in {"1", "true", "yes", "on"}


def _fallback_object_attributes(caption, reference_object_size=None):
    if reference_object_size is not None and len(reference_object_size) >= 3:
        width = float(reference_object_size[0]) / 100.0
        length = float(reference_object_size[1]) / 100.0
        height = float(reference_object_size[2]) / 100.0
    else:
        width = length = height = 1.0

    return {
        "long_caption": caption or "generated object",
        "short_caption": caption or "generated object",
        "given_caption": caption or "",
        "semantic_alignment": True,
        "name": "generated object",
        "explanation": "Object render analysis was skipped; dimensions come from the requested object size.",
        "width": width,
        "length": length,
        "height": height,
        "dimension_ordering_check": "skipped",
        "weight": max(1.0, width * length * height * 20.0),
        "scale_unit": "meter",
        "weight_unit": "kilogram",
        "pbr_parameters": {
            "explanation": "Default PBR parameters used when render analysis is skipped.",
            "metallic": 0.0,
            "roughness": 0.5,
        },
    }



class TrellisClient:
    def __init__(self, server_url=SERVER_URL):
        self.server_url = server_url.rstrip("/")

    def health_check(self):
        try:
            response = requests.get(f"{self.server_url}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("ok", False) or data.get("status") == "healthy"
        except Exception as e:
            print(f"TRELLIS health check failed: {e}", file=sys.stderr)
        return False

    def generate_model(self, input_text, seed=1, output_file="generated_model.glb"):
        print(f"Generating 3D model via TRELLIS for: '{input_text}'", file=sys.stderr)
        max_retries = 2

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Retry {attempt}/{max_retries - 1}...", file=sys.stderr)
                time.sleep(3)

            try:
                # Wait for server readiness
                for _ in range(30):
                    if self.health_check():
                        break
                    time.sleep(2)
                else:
                    print("TRELLIS server not ready after waiting", file=sys.stderr)
                    continue

                # Submit synchronous generation request (form-encoded)
                resp = requests.post(
                    f"{self.server_url}/api/text-to-3d",
                    data={"prompt": input_text},
                    timeout=300,
                )
                if resp.status_code != 200:
                    print(f"TRELLIS generation returned {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
                    continue

                result = resp.json()
                if not result.get("ok"):
                    print(f"TRELLIS generation failed: {result}", file=sys.stderr)
                    continue

                download_url = result.get("download_url")
                if not download_url:
                    print("No download URL in TRELLIS response", file=sys.stderr)
                    continue

                # Download the generated GLB
                dl = requests.get(download_url, timeout=60)
                if dl.status_code != 200:
                    print(f"Download failed: {dl.status_code}", file=sys.stderr)
                    continue

                with open(output_file, "wb") as f:
                    f.write(dl.content)
                print(f"TRELLIS model saved to {output_file} ({len(dl.content)} bytes)", file=sys.stderr)
                return True

            except Exception as e:
                print(f"TRELLIS generation error: {e}", file=sys.stderr)

        print("TRELLIS generation failed after all retries", file=sys.stderr)
        return False
  
def merge_vertices(mesh_dict):
    """
    Merge vertices of the mesh.
    Args:
        mesh_dict: A dictionary containing the mesh data.
        keys:
            "mesh": trimesh.Trimesh
            "tex_coords": {
                "vts": numpy.ndarray
                "fts": numpy.ndarray
            }
            "texture": numpy.ndarray
    Returns:
        A dictionary exactly the same as the input mesh_dict, but with the merged mesh.
    """
    import numpy as np
    from scipy.spatial import KDTree

    # print("Before merging vertices: ", mesh_dict["mesh"].vertices.shape[0], mesh_dict["mesh"].faces.shape[0], file=sys.stderr)
    
    mesh = mesh_dict["mesh"]
    tex_coords = mesh_dict["tex_coords"]
    texture = mesh_dict["texture"]
    
    # Store original mesh data for mapping
    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    original_vts = tex_coords["vts"].copy()
    original_fts = tex_coords["fts"].copy()
    
    # Merge vertices
    merged_mesh = mesh.copy()
    merged_mesh.merge_vertices(digits_vertex=6, merge_tex=True)
    # print("After merging vertices: ", merged_mesh.vertices.shape[0], merged_mesh.faces.shape[0], file=sys.stderr)
    
    # Get merged mesh data
    merged_vertices = merged_mesh.vertices
    merged_faces = merged_mesh.faces
    
    # Step 1: Find vertex mapping from merged mesh to original mesh using KDTree
    # v_ -> v
    original_tree = KDTree(original_vertices)
    distances, vertex_mapping = original_tree.query(merged_vertices)
    
    # Step 2: Find face mapping from merged mesh to original mesh
    # Unwrap faces to triangles and flatten them
    # tri_ = v_[f_.reshape(-1)].reshape(-1, 3, 3)
    merged_triangles = merged_vertices[merged_faces.reshape(-1)].reshape(-1, 3, 3)
    # tri = v[f.reshape(-1)].reshape(-1, 3, 3)  
    original_triangles = original_vertices[original_faces.reshape(-1)].reshape(-1, 3, 3)
    
    # Flatten triangles to handle all possible vertex orderings efficiently
    # For each triangle, create all 6 possible orderings: (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
    def get_all_orderings(triangles):
        """Get all 6 possible vertex orderings for each triangle"""
        orderings = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        # Shape: (n_triangles, 6, 3, 3) - 6 orderings per triangle
        all_ordered = triangles[:, None, orderings].reshape(-1, 3, 3)
        # Create mapping back to original triangle indices
        triangle_indices = np.repeat(np.arange(len(triangles)), 6)
        ordering_indices = np.tile(np.arange(6), len(triangles))
        return all_ordered, triangle_indices, ordering_indices
    
    # Get all orderings for original triangles
    original_all_ordered, orig_tri_indices, orig_order_indices = get_all_orderings(original_triangles)
    
    # Flatten triangles to points for KDTree (each triangle becomes a 9D point)
    original_flattened = original_all_ordered.reshape(-1, 9)
    merged_flattened = merged_triangles.reshape(-1, 9)
    
    # Build KDTree for triangle matching
    triangle_tree = KDTree(original_flattened)
    tri_distances, tri_mapping_indices = triangle_tree.query(merged_flattened)
    
    # Map back to original triangle indices
    face_mapping = orig_tri_indices[tri_mapping_indices]
    
    # Step 3: Update texture coordinates
    # ft_ = ft[mapping from tri_ to tri]
    new_fts = original_fts[face_mapping]
    
    # vts doesn't need to be updated (as per comment)
    new_vts = original_vts
    
    # Create the updated mesh dictionary
    updated_mesh_dict = {
        "mesh": merged_mesh,
        "tex_coords": {
            "vts": new_vts,  # Vertex texture coordinates remain the same
            "fts": new_fts  # Updated face texture indices
        },
        "texture": texture
    }
    # print("After merging vertices: ", updated_mesh_dict["mesh"].vertices.shape[0], updated_mesh_dict["mesh"].faces.shape[0], file=sys.stderr)
    
    return updated_mesh_dict


def get_lcc_mesh(mesh):
    verts = mesh.vertices
    faces = mesh.faces
    edges = mesh.edges_sorted.reshape((-1, 2))
    components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
    largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
    verts_map = components[largest_cc].reshape(-1)

    verts_map = np.sort(np.unique(verts_map))
    keep = np.zeros((verts.shape[0])).astype(np.bool_)
    keep[verts_map] = True

    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    verts_lcc = verts[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces_lcc = faces[keep_faces]

    faces_map = keep_faces

    # face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
    faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
    faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]

    return verts_lcc, faces_lcc, verts_map, faces_map

def extract_max_connected_component(mesh_dict):
    """
    Extract the maximum connected component from the mesh.
    Args:
        mesh_dict: A dictionary containing the mesh data.
        keys:
            "mesh": trimesh.Trimesh
            "tex_coords": {
                "vts": numpy.ndarray
                "fts": numpy.ndarray
            }
            "texture": numpy.ndarray

    Returns:
        A dictionary containing the mesh data of the maximum connected component.
    """
    
    mesh = mesh_dict["mesh"]
    tex_coords = mesh_dict["tex_coords"]
    texture = mesh_dict["texture"]

    # print("Before extracting LCC: ", mesh.vertices.shape[0], mesh.faces.shape[0], file=sys.stderr)

    verts_lcc, faces_lcc, verts_map, faces_map = get_lcc_mesh(mesh)

    fts = tex_coords["fts"]
    fts_new = fts[faces_map]

    mesh = trimesh.Trimesh(vertices=verts_lcc, faces=faces_lcc)

    # print("After extracting LCC: ", mesh.vertices.shape[0], mesh.faces.shape[0], file=sys.stderr)

    return {
        "mesh": mesh,
        "tex_coords": {
            "vts": tex_coords["vts"],
            "fts": fts_new
        },
        "texture": texture
    }


    
    

client = None


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def trellis_generation_enabled():
    if "SAGE_ENABLE_TRELLIS_GENERATION" in os.environ:
        return _env_bool("SAGE_ENABLE_TRELLIS_GENERATION")
    if "SAGE_DISABLE_TRELLIS" in os.environ:
        return not _env_bool("SAGE_DISABLE_TRELLIS")
    return False


def get_trellis_client():
    global client
    if client is None:
        client = TrellisClient()
        health = client.health_check()
        if health:
            print("✓ Server is running and healthy", file=sys.stderr)
            print(f"GPU available: {health.get('gpu_available', 'Unknown')}", file=sys.stderr)
        else:
            print("✗ Cannot connect to server. Make sure it's running and SSH tunnel is active.", file=sys.stderr)
    return client



def generate_model_from_text(input_text, output_path, reference_object_size=None, estimate_front=True):
    if not trellis_generation_enabled():
        raise RuntimeError(
            "TRELLIS generation is disabled by config. Set "
            "SAGE_ENABLE_TRELLIS_GENERATION=1 and SAGE_OBJECT_SOURCE=generation "
            "to use TRELLIS."
        )
    
    # Example usage
    # input_text = "A model of nightstand with two layers of drawers."
    
    # Generate just the GLB file (downloaded to local machine)
    success = get_trellis_client().generate_model(
        input_text=input_text,
        seed=random.randint(0, 1000000),
        output_file=output_path
    )
    
    if success:
        print("✓ Model generation completed successfully!", file=sys.stderr)
        mesh_dict = load_glb_advanced(output_path)
        mesh_dict = merge_vertices(mesh_dict)
        if reference_object_size is not None:
            size_text_description = f" Reference approximate size (Not the exact shape of the object, but the approximate shape of the object): {reference_object_size[0] / 100.0}m x {reference_object_size[1] / 100.0}m x {reference_object_size[2] / 100.0}m."
        else:
            size_text_description = ""
        # mesh_dict = extract_max_connected_component(mesh_dict)
        
        mesh = mesh_dict["mesh"]
        mesh_vertices = mesh.vertices.copy()
        mesh_vertices[:, 1] = mesh.vertices[:, 2]
        mesh_vertices[:, 2] = mesh.vertices[:, 1]
        mesh.vertices = mesh_vertices

        mesh.vertices[:, 0] = mesh.vertices[:, 0] - 0.5 * (mesh.vertices[:, 0].max() + mesh.vertices[:, 0].min())
        mesh.vertices[:, 1] = mesh.vertices[:, 1] - 0.5 * (mesh.vertices[:, 1].max() + mesh.vertices[:, 1].min())
        mesh.vertices[:, 2] = mesh.vertices[:, 2] - mesh.vertices[:, 2].min()

        mesh.faces = mesh.faces[:, [0, 2, 1]].copy()
        mesh_dict["tex_coords"]["fts"] = mesh_dict["tex_coords"]["fts"][:, [0, 2, 1]].copy()

        skip_render_analysis = _skip_object_render_analysis()

        if estimate_front and not skip_render_analysis:
            mesh_dict = estimate_front_from_mesh(mesh_dict)

        if skip_render_analysis:
            object_attributes = _fallback_object_attributes(input_text, reference_object_size)
            height_mean = object_attributes["height"]
            width_mean = object_attributes["width"]
            length_mean = object_attributes["length"]
        else:
            height_list = []
            width_list = []
            length_list = []
            num_inference_height = 3
            for _ in range(num_inference_height):
                object_attributes = infer_attributes_from_claude(mesh_dict, caption=input_text)
                height_list.append(object_attributes["height"])
                width_list.append(object_attributes["width"])
                length_list.append(object_attributes["length"])

            height_mean = float(np.mean(np.array(height_list)))
            object_attributes["height"] = height_mean
            width_mean = float(np.mean(np.array(width_list)))
            object_attributes["width"] = width_mean
            length_mean = float(np.mean(np.array(length_list)))
            object_attributes["length"] = length_mean

        mesh_dict["object_attributes"] = object_attributes

        scale_factor_height = height_mean / float(mesh_dict["mesh"].vertices[:, 2].max() - mesh_dict["mesh"].vertices[:, 2].min())
        # scale_factor_width = width_mean / float(mesh_dict["mesh"].vertices[:, 0].max() - mesh_dict["mesh"].vertices[:, 0].min())
        # scale_factor_length = length_mean / float(mesh_dict["mesh"].vertices[:, 1].max() - mesh_dict["mesh"].vertices[:, 1].min())

        scale_factor_xy = max(width_mean, length_mean) / max(float(mesh_dict["mesh"].vertices[:, 0].max() - mesh_dict["mesh"].vertices[:, 0].min()), float(mesh_dict["mesh"].vertices[:, 1].max() - mesh_dict["mesh"].vertices[:, 1].min()))
        # scale_factor_length = length_mean / float(mesh_dict["mesh"].vertices[:, 1].max() - mesh_dict["mesh"].vertices[:, 1].min())

        # scale_factor = float(np.mean(np.array([scale_factor_height, scale_factor_width, scale_factor_length])))

        scale_factor = np.mean([scale_factor_height, scale_factor_xy])

        mesh_dict["mesh"].vertices = mesh_dict["mesh"].vertices * scale_factor

        mesh_dict["mesh"].vertices[:, 2] = mesh_dict["mesh"].vertices[:, 2] + 0.001

        return mesh_dict
    else:
        raise Exception("Model generation failed.")



def generate_model_from_text_test_merge(input_text, output_path):
    
    # Example usage
    # input_text = "A model of nightstand with two layers of drawers."
    
    
    mesh_dict = load_glb_advanced(output_path)
    mesh_dict = merge_vertices(mesh_dict)
    mesh_dict = extract_max_connected_component(mesh_dict)
    
    mesh = mesh_dict["mesh"]
    mesh_vertices = mesh.vertices.copy()
    mesh_vertices[:, 1] = mesh.vertices[:, 2]
    mesh_vertices[:, 2] = mesh.vertices[:, 1]
    mesh.vertices = mesh_vertices

    mesh.vertices[:, 0] = mesh.vertices[:, 0] - 0.5 * (mesh.vertices[:, 0].max() + mesh.vertices[:, 0].min())
    mesh.vertices[:, 1] = mesh.vertices[:, 1] - 0.5 * (mesh.vertices[:, 1].max() + mesh.vertices[:, 1].min())
    mesh.vertices[:, 2] = mesh.vertices[:, 2] - mesh.vertices[:, 2].min()

    mesh.faces = mesh.faces[:, [0, 2, 1]].copy()
    mesh_dict["tex_coords"]["fts"] = mesh_dict["tex_coords"]["fts"][:, [0, 2, 1]].copy()

    # mesh_dict["mesh"] = mesh
    # mesh_dict["tex_coords"]["vts"][:, 1] = 1 - mesh_dict["tex_coords"]["vts"][:, 1]
    # result["tex_coords"]["vts"] = result["tex_coords"]["vts"]
    # result["tex_coords"]["fts"] = result["tex_coords"]["fts"]
    # result["texture"] = result["texture"]

    # scale_factor = infer_scale_from_reason1(mesh_dict, caption=caption)
    object_attributes = infer_attributes_from_claude(mesh_dict, caption=input_text)

    mesh_dict["object_attributes"] = object_attributes

    scale_factor = object_attributes["height"] / mesh_dict["mesh"].vertices[:, 2].max()

    mesh_dict["mesh"].vertices = mesh_dict["mesh"].vertices * scale_factor

    mesh_dict["mesh"].vertices[:, 2] = mesh_dict["mesh"].vertices[:, 2] + 0.001

    return mesh_dict

def test_generate_model_from_text():
    input_text = "A modern bed with a cushioned headboard, clean lines, and a platform base."
    output_path = os.path.join(SERVER_ROOT_DIR, "vis/objects/bed.glb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh_dict = generate_model_from_text(input_text, output_path)
    

    save_obj(
        os.path.join(SERVER_ROOT_DIR, "vis/objects/bed.obj"),
        torch.from_numpy(mesh_dict["mesh"].vertices),
        torch.from_numpy(mesh_dict["mesh"].faces),
        verts_uvs=torch.from_numpy(mesh_dict["tex_coords"]["vts"]),
        faces_uvs=torch.from_numpy(mesh_dict["tex_coords"]["fts"]),
        texture_map=torch.from_numpy(mesh_dict["texture"])
    )

def test_generate_model_from_text_dup():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    object_list = [
        "A modern white leather sofa",
        "A rustic wooden coffee table",
        "A tall floor lamp with a beige shade",
        "A potted ficus plant",
        "A red vintage armchair",
        "A marble kitchen island",
        "A bookshelf filled with colorful books",
        "A sleek office desk chair",
        "A round decorative wall mirror",
        "A blue ceramic vase",
        "A classic grand piano",
        "A modern glass dining table",
        "A set of velvet dining chairs",
        "A large persian rug",
        "A king-sized bed with white linens",
        "A bedside table with a lamp",
        "A wooden wardrobe with sliding doors",
        "A comfortable bean bag chair",
        "A minimalist tv stand",
        "A wall-mounted flat screen tv",
        "A hanging pendant light",
        "A cozy fireplace with a mantel",
        "A decorative ceiling fan",
        "A large potted palm tree",
        "A vintage record player",
        "A shelf with vinyl records",
        "A sleek espresso machine",
        "A set of kitchen knives on a block",
        "A bowl of fresh fruit",
        "A decorative throw pillow",
        "A soft wool blanket",
        "A modern art painting",
        "A framed family photograph",
        "A pair of curtains",
        "A woven laundry basket",
        "A bathroom vanity with a mirror",
        "A freestanding bathtub",
        "A towel rack with towels",
        "A shower cabin with glass doors",
        "A potted succulent garden"
    ]

    save_dir = os.path.join(SERVER_ROOT_DIR, "vis/objects/dup")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting concurrent generation of {len(object_list)} objects...", file=sys.stderr)
    
    def generate_single(idx, text):
        output_filename = f"object_{idx}_{text.replace(' ', '_')[:20]}.glb"
        output_path = os.path.join(save_dir, output_filename)
        print(f"[{idx}] Requesting: {text}", file=sys.stderr)
        try:
            result = generate_model_from_text(text, output_path)
            return idx, True, output_path
        except Exception as e:
            print(f"[{idx}] Failed: {e}", file=sys.stderr)
            return idx, False, str(e)

    # Use thread pool to generate models concurrently
    # Adjust max_workers based on expected server capacity (e.g., number of GPUs)
    max_workers = 16
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_obj = {executor.submit(generate_single, i, text): i for i, text in enumerate(object_list)}
        
        for future in as_completed(future_to_obj):
            idx = future_to_obj[future]
            try:
                idx, success, info = future.result()
                if success:
                    print(f"[{idx}] ✓ Finished: {info}", file=sys.stderr)
                else:
                    print(f"[{idx}] ✗ Failed: {info}", file=sys.stderr)
                results.append((idx, success))
            except Exception as e:
                print(f"[{idx}] ✗ Exception: {e}", file=sys.stderr)
                results.append((idx, False))
    
    success_count = sum(1 for r in results if r[1])
    print(f"\nCompleted {success_count}/{len(object_list)} generations.", file=sys.stderr)

if __name__ == "__main__":
    test_generate_model_from_text_dup()
