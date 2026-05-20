"""Load a saved layout into running Isaac Sim for interactive preview.

Usage:
    cd /home/gaok/coding/sage/server
    CONDA_ENV_NAME=sage5080 conda run -n sage5080 python load_layout_to_isaac.py --layout-id layout_ad290a83
"""
import argparse
import json
import os
import socket
import sys

ISAAC_HOST = "localhost"
ISAAC_PORT = 11323

def send_command(sock, command: dict) -> dict:
    data = json.dumps(command).encode("utf-8")
    header = len(data).to_bytes(4, "big")
    sock.sendall(header + data)
    resp_header = b""
    while len(resp_header) < 4:
        resp_header += sock.recv(4 - len(resp_header))
    resp_len = int.from_bytes(resp_header, "big")
    resp_data = b""
    while len(resp_data) < resp_len:
        resp_data += sock.recv(resp_len - len(resp_data))
    return json.loads(resp_data.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-id", required=True)
    parser.add_argument("--room-id", default=None)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    layout_dir = os.path.join(results_dir, args.layout_id)
    layout_json = os.path.join(layout_dir, f"{args.layout_id}.json")

    with open(layout_json) as f:
        layout_data = json.load(f)

    rooms = layout_data.get("rooms", [])
    if args.room_id:
        rooms = [r for r in rooms if r["id"] == args.room_id]
    if not rooms:
        print(f"No rooms found. Available: {[r['id'] for r in layout_data.get('rooms', [])]}")
        return

    room = rooms[0]
    room_id = room["id"]
    room_json_path = os.path.join(layout_dir, f"{room_id}.json")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect((ISAAC_HOST, ISAAC_PORT))
    print(f"Connected to Isaac Sim at {ISAAC_HOST}:{ISAAC_PORT}")

    cmd = {
        "type": "create_single_room_layout_scene_from_room",
        "params": {
            "scene_save_dir": layout_dir,
            "room_dict_save_path": room_json_path,
        },
    }
    print(f"Loading {room_id} ({len(room.get('objects', []))} objects)...")
    result = send_command(sock, cmd)
    print(f"Result: {result.get('status', 'unknown')}")
    if result.get("status") == "success":
        print(f"Scene loaded! You should see the room in Isaac Sim now.")
    else:
        print(f"Error: {result}")
    sock.close()


if __name__ == "__main__":
    main()
