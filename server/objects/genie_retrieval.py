"""
GenieSimAssets USD asset retriever.

Provides keyword-based retrieval of USD assets from the GenieSimAssets dataset.
USDC files are converted to trimesh via a subprocess that runs usd_to_obj.py
in a usd-core-capable Python environment.
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import trimesh


def _env(name, default):
    return os.environ.get(name, default)


GENIE_ASSETS_DIR = _env(
    "SAGE_GENIE_ASSETS_DIR",
    "/home/gaok/dataset/assets/GenieSimAssets/objects",
)
GENIE_PYTHON = _env(
    "SAGE_GENIE_PYTHON",
    "/home/gaok/anaconda3/envs/genie-scene-generator/bin/python",
)
USD_TO_OBJ_SCRIPT = os.path.join(os.path.dirname(__file__), "usd_to_obj.py")

STOPWORDS = {
    "a", "an", "the", "of", "with", "and", "or", "for", "to", "in", "on",
    "at", "by", "from", "model", "object", "3d", "small", "large", "medium",
}


def _tokenize(text):
    return {
        t for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if t not in STOPWORDS and len(t) > 1
    }


_genie_retriever = None


def get_genie_retriever():
    global _genie_retriever
    if _genie_retriever is None:
        print("Initializing GenieSimAssets retriever", file=sys.stderr)
        _genie_retriever = GenieRetriever()
    return _genie_retriever


class GenieRetriever:
    def __init__(self, assets_dir=None):
        self.assets_dir = assets_dir or GENIE_ASSETS_DIR
        self.index = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        base = self.assets_dir
        if not os.path.isdir(base):
            print(f"WARNING: GenieSimAssets directory not found: {base}", file=sys.stderr)
            return

        for category in os.listdir(base):
            cat_dir = os.path.join(base, category)
            if not os.path.isdir(cat_dir):
                continue
            if category == "benchmark":
                self._index_benchmark(cat_dir)
            else:
                self._index_flat_category(category, cat_dir)

        print(
            f"GenieRetriever indexed {len(self.index)} assets from {base}",
            file=sys.stderr,
        )

    def _index_flat_category(self, category, cat_dir):
        """Index genie/iros/edited style directories (flat: name/Aligned.usd)."""
        for name in os.listdir(cat_dir):
            item_dir = os.path.join(cat_dir, name)
            usd_path = os.path.join(item_dir, "Aligned.usd")
            if not os.path.isfile(usd_path):
                continue
            asset_id = f"{category}/{name}"
            self.index[asset_id] = {
                "usd_path": usd_path,
                "semantic_name": [name],
                "categories": [category],
                "full_description": [f"A 3D model of {name}"],
                "size_m": None,
            }

    def _index_benchmark(self, cat_dir):
        """Index benchmark style (type/variant/Aligned.usd + description.py)."""
        for obj_type in os.listdir(cat_dir):
            type_dir = os.path.join(cat_dir, obj_type)
            if not os.path.isdir(type_dir):
                continue
            for variant in os.listdir(type_dir):
                variant_dir = os.path.join(type_dir, variant)
                usd_path = os.path.join(variant_dir, "Aligned.usd")
                if not os.path.isfile(usd_path):
                    continue

                desc = self._read_description(variant_dir)
                params = self._read_object_params(variant_dir)

                asset_id = f"benchmark/{obj_type}/{variant}"
                self.index[asset_id] = {
                    "usd_path": usd_path,
                    "semantic_name": desc.get("semantic_name", [obj_type]),
                    "categories": desc.get("object_category", ["benchmark"]),
                    "full_description": desc.get("full_description", [f"A 3D model of {obj_type}"]),
                    "color": desc.get("color", ""),
                    "shape": desc.get("shape", ""),
                    "descriptive_terms": desc.get("descriptive_terms", []),
                    "size_m": params.get("size"),
                }

    @staticmethod
    def _read_description(variant_dir):
        desc_path = os.path.join(variant_dir, "description.py")
        if not os.path.isfile(desc_path):
            return {}
        try:
            with open(desc_path, "r") as f:
                # description.py is a Python dict literal
                content = f.read().strip()
            return eval(content)
        except Exception:
            return {}

    @staticmethod
    def _read_object_params(variant_dir):
        params_path = os.path.join(variant_dir, "object_parameters.json")
        if not os.path.isfile(params_path):
            return {}
        try:
            import json
            with open(params_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query, max_candidates=10):
        """
        Keyword-based retrieval.

        Returns list of (asset_id, score) tuples sorted by score descending.
        """
        query_tokens = _tokenize(query)
        scored = []
        for asset_id, meta in self.index.items():
            score = self._score(query_tokens, meta)
            if score > 0:
                scored.append((asset_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_candidates]

    def _score(self, query_tokens, meta):
        """Compute match score between query tokens and asset metadata."""
        # Build search text from all metadata fields
        text_parts = []
        for name in meta.get("semantic_name", []):
            text_parts.append(name)
        for cat in meta.get("categories", []):
            text_parts.append(cat)
        for desc in meta.get("full_description", []):
            text_parts.append(desc)
        for term in meta.get("descriptive_terms", []):
            text_parts.append(term)
        text_parts.append(meta.get("color", ""))
        text_parts.append(meta.get("shape", ""))

        index_tokens = _tokenize(" ".join(text_parts))
        overlap = len(query_tokens & index_tokens)

        # Bonus for exact semantic name match
        name_bonus = 0
        for name in meta.get("semantic_name", []):
            if str(name).lower() in " ".join(query_tokens):
                name_bonus = 3

        return overlap + name_bonus

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_object(self, asset_id):
        """
        Load a USD asset and return a dict compatible with the existing pipeline:

            {"mesh": trimesh.Trimesh, "texture": np.ndarray | None, "tex_coords": dict | None}
        """
        meta = self.index.get(asset_id)
        if meta is None:
            raise KeyError(f"Asset not found: {asset_id}")

        usd_path = meta["usd_path"]

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "mesh.obj")
            self._convert_usd_to_obj(usd_path, obj_path)
            mesh = trimesh.load(obj_path)

        # Center and bottom-align (matching Objathor loader conventions)
        if hasattr(mesh, "vertices"):
            verts = mesh.vertices
            verts[:, 0] -= 0.5 * (verts[:, 0].max() + verts[:, 0].min())
            verts[:, 1] -= 0.5 * (verts[:, 1].max() + verts[:, 1].min())
            verts[:, 2] -= verts[:, 2].min()

        # Extract UVs if available
        tex_coords = None
        texture = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
            uvs = mesh.visual.uv
            if uvs is not None and len(uvs) > 0:
                tex_coords = {
                    "vts": uvs,
                    "fts": mesh.faces.copy(),
                }

        return {
            "mesh": mesh,
            "texture": texture,
            "tex_coords": tex_coords,
        }

    def _convert_usd_to_obj(self, usd_path, obj_path):
        cmd = [
            GENIE_PYTHON,
            USD_TO_OBJ_SCRIPT,
            "--usd_path", usd_path,
            "--obj_path", obj_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"USD→OBJ conversion failed for {usd_path}: {result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"USD→OBJ conversion timed out: {usd_path}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Genie Python not found at {GENIE_PYTHON}. "
                "Set SAGE_GENIE_PYTHON to a Python interpreter with usd-core installed."
            )
