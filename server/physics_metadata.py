import json
import os
from typing import Any, Dict

from constants import RESULTS_DIR


STATIC_TYPE_KEYWORDS = (
    "table", "desk", "chair", "sofa", "cabinet", "shelf", "sideboard",
    "lamp", "stand", "bench", "counter", "wardrobe", "bookcase",
)
SMALL_DYNAMIC_TYPE_KEYWORDS = ("apple", "fruit", "book", "remote", "candle", "bottle")


def _is_wall_mounted_place_id(place_id: str) -> bool:
    place_id = (place_id or "").lower()
    return place_id == "wall" or place_id.startswith("wall")


def rule_static_object(obj) -> bool:
    if _is_wall_mounted_place_id(getattr(obj, "place_id", "")):
        return True
    if getattr(obj, "place_id", None) != "floor":
        return False

    obj_type = (getattr(obj, "type", "") or "").lower()
    dims = getattr(obj, "dimensions", None)
    footprint = float(getattr(dims, "width", 0.0)) * float(getattr(dims, "length", 0.0)) if dims else 0.0
    height = float(getattr(dims, "height", 0.0)) if dims else 0.0
    return any(keyword in obj_type for keyword in STATIC_TYPE_KEYWORDS) or footprint >= 0.20 or height >= 0.75


def rule_mass(obj) -> float:
    mass = float(getattr(obj, "mass", 1.0) or 1.0)
    dims = getattr(obj, "dimensions", None)
    if dims is None or mass > 1.0:
        return mass

    volume = max(
        float(getattr(dims, "width", 0.0))
        * float(getattr(dims, "length", 0.0))
        * float(getattr(dims, "height", 0.0)),
        0.001,
    )
    obj_type = (getattr(obj, "type", "") or "").lower()
    if rule_static_object(obj):
        return min(max(volume * 120.0, 15.0), 120.0)
    if any(keyword in obj_type for keyword in SMALL_DYNAMIC_TYPE_KEYWORDS):
        return min(max(volume * 450.0, 0.08), 2.0)
    return min(max(volume * 300.0, 0.2), 8.0)


def rule_metadata(obj) -> Dict[str, Any]:
    static = rule_static_object(obj)
    obj_type = (getattr(obj, "type", "") or "").lower()
    graspable = (
        not static
        and getattr(obj, "place_id", None) != "floor"
        or any(keyword in obj_type for keyword in SMALL_DYNAMIC_TYPE_KEYWORDS)
    )
    support_surface = any(keyword in obj_type for keyword in ("table", "desk", "shelf", "cabinet", "counter", "tray"))
    return {
        "static": static,
        "mass": round(rule_mass(obj), 4),
        "static_friction": 1.0 if static else 0.9,
        "dynamic_friction": 0.8 if static else 0.7,
        "restitution": 0.0,
        "graspable": bool(graspable),
        "support_surface": bool(support_surface),
        "collision_type": "convex_hull",
        "source": "rules",
    }


def _object_payload(obj) -> Dict[str, Any]:
    dims = getattr(obj, "dimensions", None)
    return {
        "id": getattr(obj, "id", ""),
        "type": getattr(obj, "type", ""),
        "description": getattr(obj, "description", ""),
        "place_id": getattr(obj, "place_id", ""),
        "dimensions_m": {
            "width": float(getattr(dims, "width", 0.0)) if dims else 0.0,
            "length": float(getattr(dims, "length", 0.0)) if dims else 0.0,
            "height": float(getattr(dims, "height", 0.0)) if dims else 0.0,
        },
        "initial_mass_kg": float(getattr(obj, "mass", 1.0) or 1.0),
    }


def _clamp(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, lower), upper)


def _sanitize_metadata(obj, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    fallback = rule_metadata(obj)
    if not isinstance(metadata, dict):
        return fallback

    obj_type = (getattr(obj, "type", "") or "").lower()
    static = bool(metadata.get("static", fallback["static"]))
    if rule_static_object(obj):
        static = True

    if static:
        min_mass, max_mass = 15.0, 200.0
    elif any(keyword in obj_type for keyword in SMALL_DYNAMIC_TYPE_KEYWORDS):
        min_mass, max_mass = 0.05, 3.0
    else:
        min_mass, max_mass = 0.1, 20.0

    mass = _clamp(metadata.get("mass_kg", metadata.get("mass", fallback["mass"])), fallback["mass"], min_mass, max_mass)
    return {
        "static": static,
        "mass": round(mass, 4),
        "static_friction": _clamp(metadata.get("static_friction"), fallback["static_friction"], 0.0, 2.0),
        "dynamic_friction": _clamp(metadata.get("dynamic_friction"), fallback["dynamic_friction"], 0.0, 2.0),
        "restitution": _clamp(metadata.get("restitution"), fallback["restitution"], 0.0, 1.0),
        "graspable": bool(metadata.get("graspable", fallback["graspable"])) and not static,
        "support_surface": bool(metadata.get("support_surface", fallback["support_surface"])),
        "collision_type": metadata.get("collision_type", fallback["collision_type"]),
        "source": metadata.get("source", "llm"),
    }


def _response_text(response) -> str:
    if hasattr(response, "content") and response.content:
        return response.content[0].text
    if hasattr(response, "choices") and response.choices:
        return response.choices[0].message.content
    return str(response)


def _call_llm_for_metadata(layout) -> Dict[str, Dict[str, Any]]:
    from utils import extract_json_from_response
    from vlm import call_vlm

    objects = []
    for room in layout.rooms:
        for obj in room.objects:
            payload = _object_payload(obj)
            payload["room_type"] = room.room_type
            objects.append(payload)

    prompt = f"""You generate stable physics metadata for robot simulation assets.
Return ONLY valid JSON with an "objects" list. Each item must include:
id, mass_kg, static, graspable, support_surface, static_friction, dynamic_friction, restitution, collision_type.

Guidelines:
- Large furniture and support furniture should usually be static or heavy.
- Small task objects should be dynamic and graspable when reasonable.
- Restitution should usually be 0.0 for household objects.
- Use SI units and realistic household masses.
- collision_type should be one of: box, convex_hull, mesh, none.

Objects:
{json.dumps(objects, indent=2)}
"""
    response = call_vlm(
        vlm_type=os.environ.get("SAGE_PHYSICS_METADATA_LLM_TYPE", "openai"),
        model=os.environ.get("SAGE_PHYSICS_METADATA_MODEL", "openai/gpt-oss-120b"),
        max_tokens=int(os.environ.get("SAGE_PHYSICS_METADATA_MAX_TOKENS", "6000")),
        temperature=float(os.environ.get("SAGE_PHYSICS_METADATA_TEMPERATURE", "0.1")),
        messages=[{"role": "user", "content": prompt}],
        max_retries=1,
    )
    text = extract_json_from_response(_response_text(response))
    data = json.loads(text)
    return {item["id"]: item for item in data.get("objects", []) if "id" in item}


def get_layout_physics_metadata(layout) -> Dict[str, Dict[str, Any]]:
    metadata = {obj.id: rule_metadata(obj) for room in layout.rooms for obj in room.objects}
    if os.environ.get("SAGE_USE_LLM_PHYSICS_METADATA", "").lower() in {"0", "false", "no"}:
        return metadata

    cache_path = os.path.join(RESULTS_DIR, layout.id, "physics_metadata_cache.json")
    try:
        if os.path.exists(cache_path) and os.environ.get("SAGE_REFRESH_LLM_PHYSICS_METADATA", "").lower() not in {"1", "true", "yes"}:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            llm_metadata = cached.get("objects", cached)
        else:
            llm_metadata = _call_llm_for_metadata(layout)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"objects": llm_metadata}, f, indent=4)

        obj_by_id = {obj.id: obj for room in layout.rooms for obj in room.objects}
        for obj_id, obj in obj_by_id.items():
            metadata[obj_id] = _sanitize_metadata(obj, llm_metadata.get(obj_id))
    except Exception as exc:
        print(f"Warning: LLM physics metadata failed, using rule fallback: {exc}", file=os.sys.stderr)
    return metadata
