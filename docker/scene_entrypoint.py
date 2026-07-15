#!/usr/bin/env python3
"""One-shot SAGE job: generate through external Isaac, upload artifacts, exit."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def isaac_endpoint(value: str) -> tuple[str, int]:
    parsed = urlparse(value if "://" in value else f"tcp://{value}")
    if not parsed.hostname:
        raise RuntimeError(f"invalid ISAAC_SIM_URL: {value!r}")
    return parsed.hostname, parsed.port or 11323


def choose_artifacts(layout_dir: Path, layout_id: str) -> tuple[Path, Path]:
    usd_candidates = [
        layout_dir / f"{layout_id}_view.usd",
        layout_dir / f"{layout_id}.usd",
    ]
    usd_candidates.extend(sorted(layout_dir.glob("*.usd")))
    usd = next((path for path in usd_candidates if path.is_file()), None)

    preview_dir = layout_dir / "preview"
    thumb_candidates = sorted(preview_dir.glob(f"{layout_id}_full_view_*.png"))
    if not thumb_candidates:
        thumb_candidates = sorted(preview_dir.glob("*.png"))
    thumb = thumb_candidates[0] if thumb_candidates else None

    if usd is None:
        raise RuntimeError(f"no USD artifact found under {layout_dir}")
    if thumb is None:
        raise RuntimeError(f"no PNG preview found under {preview_dir}")
    return usd, thumb


def upload_with_mc(files: list[tuple[Path, str]]) -> None:
    mc = Path(os.environ.get("MINIO_MC_PATH", "/tools/mc"))
    if not mc.is_file():
        raise RuntimeError(f"MinIO client not found: {mc}")

    endpoint = required("MINIO_ENDPOINT")
    access_key = required("MINIO_ACCESS_KEY")
    secret_key = required("MINIO_SECRET_KEY")
    bucket = required("MINIO_BUCKET")
    alias = "scene-job"
    subprocess.run(
        [str(mc), "alias", "set", alias, endpoint, access_key, secret_key],
        check=True,
    )
    for source, object_name in files:
        subprocess.run(
            [str(mc), "cp", str(source), f"{alias}/{bucket}/{object_name}"],
            check=True,
        )


def main() -> int:
    # The external Isaac process and this job commonly run under different
    # UIDs while sharing the results bind mount. Ensure directories and files
    # created by the job remain writable by Isaac during the same run.
    os.umask(0)
    prompt = required("PROMPT")
    isaac_url = required("ISAAC_SIM_URL")
    scene_name = required("SCENE_NAME")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", scene_name):
        raise RuntimeError(
            "SCENE_NAME must be 1-128 characters using letters, digits, '.', '_' or '-'"
        )
    # Validate upload configuration before starting an expensive generation.
    for variable in (
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
    ):
        required(variable)
    prefix = os.environ.get("MINIO_OBJECT_PREFIX", "").strip("/")
    root = Path(os.environ.get("SAGE_ROOT", "/app")).resolve()
    shared_root = Path(os.environ.get("SAGE_SHARED_ROOT", "/shared")).resolve()
    objathor_root = Path(
        os.environ.get("SAGE_OBJATHOR_ROOT", str(shared_root / "objathor"))
    ).resolve()
    results_root = Path(
        os.environ.get("SAGE_RESULTS_DIR", str(shared_root / "results"))
    ).resolve()
    output_dir = Path(os.environ.get("SCENE_OUTPUT_DIR", "/tmp/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    if not objathor_root.is_dir():
        raise RuntimeError(
            f"Objathor data not found: {objathor_root}; mount it at the same absolute "
            "path in this container and on the external Isaac Sim host"
        )

    host, port = isaac_endpoint(isaac_url)
    os.environ["SAGE_ISAAC_HOST"] = host
    os.environ["SAGE_ISAAC_MCP_PORT"] = str(port)
    os.environ["SAGE_RESULTS_DIR"] = str(results_root)
    # Do not force HOLODECK_BASE_DATA_DIR: component-wise Objathor downloads
    # keep the complete door bundle under objathor/doors/.objathor-assets while
    # room materials live under objathor/<version>. The selectors discover both.
    os.environ.setdefault("OBJATHOR_FEATURES_DIR", str(objathor_root / "features"))
    os.environ.setdefault(
        "OBJATHOR_ANNOTATIONS_PATH", str(objathor_root / "annotations.json.gz")
    )
    os.environ.setdefault("OBJATHOR_ASSETS_DIR", str(objathor_root / "assets" / "assets"))
    os.environ.setdefault("SAGE_DISABLE_TRELLIS", "1")
    os.environ.setdefault("SAGE_OBJATHOR_RETRIEVAL_MODE", "embedding")
    os.environ.setdefault("SAGE_OBJECT_SOURCE", "objaverse")
    os.environ.setdefault("SAGE_DISABLE_MATFUSE", "1")

    print(f"[scene-gen] PROMPT={prompt}", flush=True)
    print(f"[scene-gen] ISAAC_SIM_URL={isaac_url}", flush=True)
    print(f"[scene-gen] SCENE_NAME={scene_name}", flush=True)
    print(f"[scene-gen] MINIO_OBJECT_PREFIX={prefix}", flush=True)
    print(f"[scene-gen] SAGE_RESULTS_DIR={results_root}", flush=True)

    with socket.create_connection((host, port), timeout=15):
        print(f"[scene-gen] external Isaac MCP reachable at {host}:{port}", flush=True)

    client_dir = root / "client"
    command = [
        sys.executable,
        "client_generation_room_desc.py",
        "--room_desc",
        prompt,
        "--server_paths",
        "../server/layout_wo_robot.py",
        "--max_tool_calls",
        os.environ.get("SAGE_MAX_TOOL_CALLS", "40"),
    ]
    layout_id = ""
    process = subprocess.Popen(
        command,
        cwd=client_dir,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = re.search(r"Layout ID:\s*(layout_[A-Za-z0-9_-]+)", line)
        if match:
            layout_id = match.group(1)
    if process.wait() != 0:
        raise RuntimeError(f"scene generation failed with exit code {process.returncode}")

    if not layout_id:
        candidates = [path for path in results_root.glob("layout_*") if path.is_dir()]
        if not candidates:
            raise RuntimeError("generation completed but no layout result directory was found")
        layout_id = max(candidates, key=lambda path: path.stat().st_mtime).name

    layout_dir = results_root / layout_id
    usd_source, thumb_source = choose_artifacts(layout_dir, layout_id)
    usd_output = output_dir / f"{scene_name}.usd"
    thumb_output = output_dir / "thumb.png"
    shutil.copy2(usd_source, usd_output)
    shutil.copy2(thumb_source, thumb_output)

    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scene_name": scene_name,
                "layout_id": layout_id,
                "usd": usd_output.name,
                "thumbnail": thumb_output.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    object_root = f"{prefix}/" if prefix else ""
    upload_with_mc(
        [
            (usd_output, f"{object_root}{scene_name}.usd"),
            (thumb_output, f"{object_root}thumb.png"),
            (manifest, f"{object_root}manifest.json"),
        ]
    )
    print("[scene-gen] artifacts uploaded", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[scene-gen] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
