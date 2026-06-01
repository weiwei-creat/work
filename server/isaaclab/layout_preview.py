import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import sys

server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from models import Dimensions, Door, Euler, FloorPlan, Object, Point3D, Room, Wall, Window


RESULTS_DIR = os.path.join(server_dir, "results")


def dict_to_floor_plan(layout_data: dict) -> FloorPlan:
    rooms = [dict_to_room(room_data) for room_data in layout_data["rooms"]]
    return FloorPlan(
        id=layout_data["id"],
        rooms=rooms,
        total_area=layout_data["total_area"],
        building_style=layout_data["building_style"],
        description=layout_data["description"],
        created_from_text=layout_data["created_from_text"],
        policy_analysis=layout_data.get("policy_analysis"),
    )


def dict_to_room(room_data: dict) -> Room:
    return Room(
        id=room_data["id"],
        room_type=room_data["room_type"],
        position=dict_to_point(room_data["position"]),
        dimensions=dict_to_dimensions(room_data["dimensions"]),
        walls=[dict_to_wall(wall_data) for wall_data in room_data.get("walls", [])],
        doors=[dict_to_door(door_data) for door_data in room_data.get("doors", [])],
        objects=[dict_to_object(object_data) for object_data in room_data.get("objects", [])],
        windows=[dict_to_window(window_data) for window_data in room_data.get("windows", [])],
        floor_material=room_data.get("floor_material", "hardwood"),
        ceiling_height=room_data.get("ceiling_height", 2.7),
    )


def dict_to_point(point_data: dict) -> Point3D:
    return Point3D(x=point_data["x"], y=point_data["y"], z=point_data["z"])


def dict_to_dimensions(dimensions_data: dict) -> Dimensions:
    return Dimensions(
        width=dimensions_data["width"],
        length=dimensions_data["length"],
        height=dimensions_data["height"],
    )


def dict_to_wall(wall_data: dict) -> Wall:
    return Wall(
        id=wall_data["id"],
        start_point=dict_to_point(wall_data["start_point"]),
        end_point=dict_to_point(wall_data["end_point"]),
        height=wall_data["height"],
        thickness=wall_data.get("thickness", 0.1),
        material=wall_data.get("material", "drywall"),
    )


def dict_to_door(door_data: dict) -> Door:
    return Door(
        id=door_data["id"],
        wall_id=door_data["wall_id"],
        position_on_wall=door_data["position_on_wall"],
        width=door_data["width"],
        height=door_data["height"],
        door_type=door_data.get("door_type", "standard"),
        opens_inward=door_data.get("opens_inward", True),
        opening=door_data.get("opening", False),
        door_material=door_data.get("door_material", "standard"),
    )


def dict_to_window(window_data: dict) -> Window:
    return Window(
        id=window_data["id"],
        wall_id=window_data["wall_id"],
        position_on_wall=window_data["position_on_wall"],
        width=window_data["width"],
        height=window_data["height"],
        sill_height=window_data["sill_height"],
        window_type=window_data.get("window_type", "standard"),
        window_material=window_data.get("window_material", "standard"),
    )


def dict_to_object(object_data: dict) -> Object:
    return Object(
        id=object_data["id"],
        room_id=object_data["room_id"],
        type=object_data["type"],
        description=object_data["description"],
        position=dict_to_point(object_data["position"]),
        rotation=Euler(
            x=object_data["rotation"]["x"],
            y=object_data["rotation"]["y"],
            z=object_data["rotation"]["z"],
        ),
        dimensions=dict_to_dimensions(object_data["dimensions"]),
        source=object_data["source"],
        source_id=object_data["source_id"],
        place_id=object_data["place_id"],
        place_guidance=object_data.get("place_guidance", "Standard placement for the object"),
        placement_constraints=object_data.get("placement_constraints"),
        mass=object_data.get("mass", 1.0),
    )


def slurm_job_id_to_port(job_id, port_start=8080, port_end=40000):
    job_id_str = str(job_id)
    hash_int = int(hashlib.md5(job_id_str.encode()).hexdigest(), 16)
    return port_start + (hash_int % (port_end - port_start + 1))


def send_isaac_command(command: dict, timeout: float = 300.0) -> dict:
    host = os.environ.get("ISAAC_MCP_HOST", "localhost")
    port = int(os.environ.get("ISAAC_MCP_PORT", slurm_job_id_to_port(os.environ.get("SLURM_JOB_ID"))))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    try:
        sock.sendall(json.dumps(command).encode("utf-8"))
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            try:
                return json.loads(b"".join(chunks).decode("utf-8"))
            except json.JSONDecodeError:
                continue
        raise RuntimeError("Connection closed before a complete Isaac response was received")
    finally:
        sock.close()


def generate_isaac_preview(layout_id: str, layout: FloorPlan, resolution: int = 1024):
    scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
    preview_save_dir = os.path.join(scene_save_dir, "preview")
    if os.path.isdir(preview_save_dir):
        shutil.rmtree(preview_save_dir)
    os.makedirs(preview_save_dir, exist_ok=True)

    for room in layout.rooms:
        load_response = send_isaac_command(
            {
                "type": "create_single_room_layout_scene",
                "params": {
                    "scene_save_dir": scene_save_dir,
                    "room_id": room.id,
                },
            }
        )
        if load_response.get("status") != "success":
            raise RuntimeError(f"Isaac failed to load room {room.id}: {load_response}")

        render_response = send_isaac_command(
            {
                "type": "render_room_preview",
                "params": {
                    "scene_save_dir": scene_save_dir,
                    "room_id": room.id,
                    "resolution": resolution,
                    "num_views": 4,
                },
            }
        )
        if render_response.get("status") != "success":
            raise RuntimeError(f"Isaac failed to render room {room.id}: {render_response}")


async def generate_preview(layout_id: str, resolution: int):
    layout_save_path = os.path.join(RESULTS_DIR, layout_id, f"{layout_id}.json")
    with open(layout_save_path, "r") as f:
        layout_data = json.load(f)
    layout = dict_to_floor_plan(layout_data)

    generate_isaac_preview(layout_id, layout, resolution=resolution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()
    asyncio.run(generate_preview(args.layout_id, resolution=args.resolution))
