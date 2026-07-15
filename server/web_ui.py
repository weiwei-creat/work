#!/usr/bin/env python3
"""Lightweight web console for SAGE room generation."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


SAGE_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = SAGE_ROOT / "server"
CLIENT_ROOT = SAGE_ROOT / "client"
RESULTS_ROOT = SERVER_ROOT / "results"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Generation/display split: the managed Isaac MCP kit runs HEADLESS during
# generation (faster, no GPU contention). When a generation finishes, the kit
# is stopped and a separate GUI display process is launched on the workstation
# display — a lit scene viewer for plain scenes, or the robot-task runner for
# robot generations. The next /api/generate closes the display process and
# brings the headless kit back automatically.
# SAGE_ISAAC_WINDOW=1 restores the old windowed-kit behavior (debug only).
_ISAAC_FLAGS = (
    "--experience isaacsim.exp.base.kit "
    f"--ext-folder {shlex.quote(str(SERVER_ROOT / 'isaacsim'))} "
    "--enable isaac.sim.mcp_extension"
)
_ISAAC_DISPLAY = os.environ.get("SAGE_ISAAC_DISPLAY", ":0")
_want_isaac_window = os.environ.get("SAGE_ISAAC_WINDOW", "0").strip().lower() not in {"0", "false", "no"}
_display_available = Path(f"/tmp/.X11-unix/X{_ISAAC_DISPLAY.lstrip(':').split('.')[0]}").exists()
if _want_isaac_window and _display_available:
    DEFAULT_ISAAC_CMD = f"DISPLAY={_ISAAC_DISPLAY} ./client/isaac_sim_conda.sh {_ISAAC_FLAGS}"
else:
    DEFAULT_ISAAC_CMD = f"./client/isaac_sim_conda.sh --no-window {_ISAAC_FLAGS}"

# GUI display processes (post-generation). Run with Isaac's own python.sh so
# they boot an independent SimulationApp — the headless kit is stopped first.
_isaac_root = Path(os.environ.get("ISAAC_SIM_PATH", "/home/ubuntu/isaac"))
_isaac_python_candidates = (
    _isaac_root / "python.sh",
    _isaac_root / "_build/linux-x86_64/release/python.sh",
)
ISAAC_PYTHON_SH = os.environ.get(
    "SAGE_ISAAC_PYTHON_SH",
    str(next((path for path in _isaac_python_candidates if path.exists()), _isaac_python_candidates[0])),
)
SCENE_VIEWER_SCRIPT = "server/isaacsim/scene_viewer.py"
G1_WALK_SCRIPT = "server/isaacsim/isaac.sim.mcp_extension/examples/run_g1_walk.sh"
# Kinematic Franka pick-and-place visualizer (plain Isaac 5.x SimulationApp).
# NOTE: the repo's IsaacLab fork (0.30.x / Isaac 4.2-era) does NOT load against
# Isaac Sim 5.1, so the original data_generation_* runners cannot be used here.
FRANKA_VIZ_SCRIPT = "server/isaacsim/isaac.sim.mcp_extension/examples/franka_task_render.py"
_trellis_root = os.environ.get("TRELLIS_ROOT", str(SAGE_ROOT.parent / "TRELLIS"))
_trellis_env = os.environ.get("TRELLIS_CONDA_ENV", "trellis")
DEFAULT_TRELLIS_CMD = (
    f"bash scripts/start_trellis_server.sh 8080 "
    f"{shlex.quote(_trellis_root)} {shlex.quote(_trellis_env)}"
)
ISAAC_HOST = os.environ.get("SAGE_ISAAC_HOST", "localhost")
DEFAULT_GENERATION_PYTHON = os.environ.get(
    "SAGE_GENERATION_PYTHON",
    sys.executable,
)
DEFAULT_MAX_TOOL_CALLS = int(os.environ.get("SAGE_MAX_TOOL_CALLS", "40"))


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SAGE 生成控制台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #e9edf2;
      --panel: #ffffff;
      --panel-soft: #f6f8fb;
      --line: #d8dee7;
      --text: #17212b;
      --muted: #647282;
      --brand: #1f7a8c;
      --brand-dark: #135765;
      --accent: #c98b2f;
      --ok: #1d7f55;
      --warn: #a76400;
      --bad: #a33b3b;
      --bubble-user: #d9f0f4;
      --bubble-system: #f2f5f8;
      --bubble-model: #fff7e7;
      --bubble-tool: #ecf7ef;
      --shadow: 0 18px 46px rgba(18, 31, 43, 0.12);
      --soft-shadow: 0 8px 22px rgba(18, 31, 43, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(135deg, rgba(31,122,140,0.10), transparent 28%),
        linear-gradient(315deg, rgba(201,139,47,0.10), transparent 32%),
        var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
    }

    .app {
      display: grid;
      grid-template-columns: minmax(420px, 0.92fr) minmax(480px, 1.08fr);
      grid-template-rows: minmax(0, 1fr) minmax(132px, 224px);
      gap: 14px;
      height: calc(100vh - 28px);
      margin: 14px;
      background: transparent;
    }

    .panel {
      background: var(--panel);
      border: 1px solid rgba(216,222,231,0.92);
      border-radius: 12px;
      box-shadow: var(--shadow);
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .left { }
    .right { }

    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
    }

    h1, h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 720;
      letter-spacing: 0;
    }

    .sub {
      margin-top: 2px;
      font-size: 12px;
      color: var(--muted);
    }

    .brand-lockup {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 38px;
      height: 38px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: #e9f5f7;
      color: var(--brand-dark);
      border: 1px solid #c6e3e9;
      font-weight: 800;
      flex: 0 0 auto;
    }

    .status-row {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .timer-pill {
      display: none;
      align-items: center;
      gap: 6px;
      padding: 4px 11px;
      border-radius: 999px;
      background: #eef2f5;
      color: var(--muted);
      font-size: 13px;
      font-weight: 720;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .timer-pill.show { display: inline-flex; }
    .timer-pill.running { background: #fff5e5; color: var(--warn); }
    .timer-pill.done { background: #eaf6f0; color: var(--ok); }

    .switch-line {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
      min-width: 120px;
    }

    .switch-text {
      min-width: 56px;
      color: var(--text);
      font-weight: 620;
    }

    .spinner {
      display: none;
      width: 14px;
      height: 14px;
      border: 2px solid #c8d0d8;
      border-top-color: var(--brand);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    .switch-line.loading .spinner { display: inline-block; }
    .switch-line.loading .switch { opacity: 0.72; }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .switch {
      width: 46px;
      height: 24px;
      position: relative;
      display: inline-block;
    }

    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute;
      cursor: pointer;
      inset: 0;
      background: #c8d0d8;
      border-radius: 999px;
      transition: 0.18s;
      border: 1px solid #b8c1ca;
    }
    .slider::before {
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 2px;
      top: 2px;
      background: white;
      border-radius: 50%;
      transition: 0.18s;
      box-shadow: 0 1px 4px rgba(0,0,0,0.22);
    }
    .switch input:checked + .slider {
      background: var(--brand);
      border-color: var(--brand);
    }
    .switch input:checked + .slider::before { transform: translateX(22px); }

    .composer {
      padding: 14px 18px 16px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 10px;
      background: rgba(255,255,255,0.96);
    }

    textarea {
      width: 100%;
      min-height: 78px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      line-height: 1.45;
      color: var(--text);
      outline: none;
    }
    textarea:focus { border-color: var(--brand); box-shadow: 0 0 0 3px rgba(37,111,131,0.12); }

    .task-opts {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .opt-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12.5px;
      color: var(--muted);
      white-space: nowrap;
    }
    .opt-label select, .opt-label input {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 6px 9px;
      font: inherit;
      font-size: 13px;
      color: var(--text);
      background: var(--panel);
      outline: none;
    }
    .opt-label select:focus, .opt-label input:focus { border-color: var(--brand); box-shadow: 0 0 0 3px rgba(37,111,131,0.12); }
    .opt-label input { width: 140px; }

    .controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 680;
      cursor: pointer;
      background: var(--brand);
      color: white;
    }
    button:hover { background: var(--brand-dark); }
    button:disabled {
      cursor: not-allowed;
      background: #b9c2cb;
      color: #eef1f4;
    }
    .secondary {
      background: #edf2f4;
      color: var(--brand-dark);
      border: 1px solid #ccdade;
    }
    .secondary:hover { background: #e1ebee; }
    .danger {
      background: #fbeaea;
      color: var(--bad);
      border: 1px solid #e6b5b5;
    }
    .danger:hover { background: #f6dada; }
    .danger:disabled { background: #f0f2f4; color: #b9c2cb; border-color: #dde2e7; }

    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    .chat {
      flex: 1;
      min-height: 0;
      height: 100%;
      overflow-y: auto;
      overscroll-behavior: contain;
      scroll-behavior: smooth;
      scrollbar-width: thin;
      scrollbar-color: #b9c2cb #edf2f4;
      padding: 20px 22px 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background:
        linear-gradient(rgba(251,252,253,0.92), rgba(251,252,253,0.92)),
        radial-gradient(circle at 18px 18px, rgba(100,114,130,0.16) 1px, transparent 1.5px);
      background-size: auto, 28px 28px;
    }
    .chat::-webkit-scrollbar { width: 10px; }
    .chat::-webkit-scrollbar-track { background: #edf2f4; border-radius: 8px; }
    .chat::-webkit-scrollbar-thumb { background: #b9c2cb; border-radius: 8px; border: 2px solid #edf2f4; min-height: 40px; }
    .chat::-webkit-scrollbar-thumb:hover { background: #8f99a5; }

    .msg {
      flex: 0 0 auto;
      max-width: min(82%, 680px);
      border-radius: 8px;
      border: 1px solid var(--line);
      box-shadow: var(--soft-shadow);
      overflow: hidden;
      position: relative;
    }
    .msg.local { align-self: flex-end; background: var(--bubble-user); border-color: #b7dde5; }
    .msg.user { align-self: flex-end; background: var(--bubble-user); border-color: #b7dde5; }
    .msg.server { align-self: flex-start; background: var(--bubble-system); }
    .msg.system { align-self: flex-start; background: var(--bubble-system); }
    .msg.model { align-self: flex-start; background: var(--bubble-model); }
    .msg.tool { align-self: flex-start; background: var(--bubble-tool); }
    .msg.error { align-self: flex-start; background: #fff0f0; border-color: #e6b5b5; }
    .msg.image { align-self: flex-start; background: #ffffff; border-color: #d9e1e8; }

    .msg::after {
      content: "";
      position: absolute;
      bottom: 10px;
      width: 10px;
      height: 10px;
      background: inherit;
      border-bottom: 1px solid var(--line);
      transform: rotate(45deg);
    }
    .msg.local::after, .msg.user::after {
      right: -6px;
      border-right: 1px solid #b7dde5;
    }
    .msg.server::after, .msg.system::after, .msg.model::after, .msg.tool::after, .msg.error::after {
      left: -6px;
      border-left: 1px solid var(--line);
    }

    .msg-head {
      padding: 8px 11px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      border-bottom: 1px solid rgba(0,0,0,0.06);
    }
    .msg-body {
      padding: 11px 12px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-size: 13.5px;
    }

    .chat-image {
      display: block;
      max-width: min(360px, 100%);
      max-height: 320px;
      object-fit: contain;
      border-radius: 8px;
      background: #edf2f5;
      box-shadow: 0 4px 14px rgba(18,31,43,0.12);
    }

    .image-caption {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-word;
    }

    .panel-title {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 720;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }

    .empty {
      color: var(--muted);
      display: grid;
      place-items: center;
      min-height: 220px;
      padding: 24px;
      text-align: center;
      line-height: 1.5;
    }

    .stage-list {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }

    .stage {
      display: grid;
      grid-template-columns: 18px 1fr auto;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fbfcfd;
      font-size: 12px;
      color: var(--muted);
    }

    .stage-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #c4ccd6;
      justify-self: center;
    }

    .stage.done .stage-dot { background: var(--ok); }
    .stage.active .stage-dot { background: var(--accent); box-shadow: 0 0 0 4px rgba(201,139,47,0.16); }
    .stage strong { color: var(--text); font-weight: 680; }

    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      background: #eef2f5;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.ok { color: var(--ok); background: #eaf6f0; }
    .pill.warn { color: var(--warn); background: #fff5e5; }
    .pill.bad { color: var(--bad); background: #fff0f0; }

    /* ---------- right results panel ---------- */
    .results {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      scrollbar-width: thin;
      scrollbar-color: #b9c2cb #edf2f4;
      padding: 18px 20px 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      background:
        linear-gradient(rgba(250,251,253,0.96), rgba(250,251,253,0.96)),
        radial-gradient(circle at 16px 16px, rgba(100,114,130,0.10) 1px, transparent 1.5px);
      background-size: auto, 30px 30px;
    }
    .results::-webkit-scrollbar { width: 10px; }
    .results::-webkit-scrollbar-thumb { background: #b9c2cb; border-radius: 8px; border: 2px solid #edf2f4; }

    .layout-chip {
      display: none;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e9f5f7;
      border: 1px solid #c6e3e9;
      color: var(--brand-dark);
      font-size: 12px;
      font-weight: 640;
      cursor: pointer;
    }
    .layout-chip.show { display: inline-flex; }
    .layout-chip:hover { background: #ddeef1; }

    /* tool-call progress bar */
    .pbar-wrap {
      display: none;
      flex-direction: column;
      gap: 6px;
      padding: 2px 2px 0;
    }
    .pbar-wrap.show { display: flex; }
    .pbar-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      font-size: 12px;
      color: var(--muted);
    }
    .pbar-head .pcount { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
    .pbar-head .peta { font-variant-numeric: tabular-nums; }
    .pbar {
      height: 9px;
      border-radius: 999px;
      background: #e3e9ef;
      overflow: hidden;
      border: 1px solid var(--line);
    }
    .pbar-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--brand), var(--accent));
      transition: width 0.4s ease;
    }
    .pbar-fill.done { background: var(--ok); }

    /* progress tracker */
    .progress {
      display: flex;
      align-items: flex-start;
      gap: 0;
      padding: 4px 2px 2px;
    }
    .pstep {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      position: relative;
      color: var(--muted);
      font-size: 11.5px;
      text-align: center;
    }
    .pstep:not(:last-child)::after {
      content: "";
      position: absolute;
      top: 11px;
      left: 50%;
      width: 100%;
      height: 2px;
      background: #d8dee7;
      z-index: 0;
    }
    .pstep.done:not(:last-child)::after { background: var(--ok); }
    .pdot {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: #fff;
      border: 2px solid #cdd5de;
      display: grid;
      place-items: center;
      font-size: 12px;
      color: var(--muted);
      z-index: 1;
      transition: 0.2s;
    }
    .pstep.active .pdot { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 0 4px rgba(201,139,47,0.16); }
    .pstep.done .pdot { border-color: var(--ok); background: var(--ok); color: #fff; }
    .pstep.active .plabel { color: var(--text); font-weight: 640; }

    /* hero + gallery */
    .hero-wrap { display: none; flex-direction: column; gap: 10px; }
    .hero-wrap.show { display: flex; }
    .hero {
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 12px;
      object-fit: contain;
      background: #11161c;
      border: 1px solid var(--line);
      box-shadow: var(--soft-shadow);
      cursor: zoom-in;
    }
    .hero-cap { color: var(--muted); font-size: 12px; text-align: center; word-break: break-word; }
    .thumbs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .thumb {
      width: 84px;
      height: 84px;
      border-radius: 8px;
      object-fit: cover;
      background: #edf2f5;
      border: 2px solid transparent;
      cursor: pointer;
      transition: 0.15s;
    }
    .thumb:hover { border-color: #c6e3e9; }
    .thumb.active { border-color: var(--brand); }

    .section-label {
      font-size: 12px;
      font-weight: 680;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    /* summary card */
    .summary {
      display: none;
      flex-direction: column;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      background: linear-gradient(180deg, #ffffff, #f8fbfc);
      box-shadow: var(--soft-shadow);
    }
    .summary.show { display: flex; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fbfcfd;
    }
    .stat .k { font-size: 11px; color: var(--muted); }
    .stat .v { font-size: 20px; font-weight: 760; color: var(--text); margin-top: 2px; }
    .stat .v small { font-size: 12px; font-weight: 600; color: var(--muted); }

    .results-empty {
      flex: 1;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      gap: 12px;
      padding: 32px;
    }
    .results-empty .big {
      width: 56px; height: 56px;
      display: grid; place-items: center;
      border-radius: 14px;
      background: #eef5f7;
      border: 1px solid #d3e6ea;
      font-size: 26px;
    }
    .results-empty.hide { display: none; }

    /* lightbox */
    .lightbox {
      position: fixed;
      inset: 0;
      background: rgba(10,16,22,0.86);
      display: none;
      place-items: center;
      z-index: 50;
      padding: 32px;
      cursor: zoom-out;
    }
    .lightbox.show { display: grid; }
    .lightbox img { max-width: 94vw; max-height: 92vh; border-radius: 10px; box-shadow: 0 24px 60px rgba(0,0,0,0.5); }

    /* ---------- bottom terminal log ---------- */
    .logbar {
      grid-column: 1 / -1;
      min-height: 0;
    }
    .logbar .panel-title {
      background: rgba(255,255,255,0.92);
    }
    .term {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 10px 14px 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.55;
      background: #0f1620;
      color: #cdd6e0;
      scrollbar-width: thin;
      scrollbar-color: #34465a #0f1620;
    }
    .term::-webkit-scrollbar { width: 10px; }
    .term::-webkit-scrollbar-track { background: #0f1620; }
    .term::-webkit-scrollbar-thumb { background: #2c3a4a; border-radius: 8px; border: 2px solid #0f1620; }
    .term::-webkit-scrollbar-thumb:hover { background: #3c5066; }
    .term-line {
      white-space: pre-wrap;
      word-break: break-word;
    }
    .term-line.err { color: #ff9c9c; }
    .term-empty { color: #5d6b7a; }

    @media (max-width: 980px) {
      body { overflow: auto; }
      .app { display: block; height: auto; min-height: 100vh; }
      .panel { min-height: 70vh; margin-bottom: 14px; }
      .logbar.panel { min-height: 240px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="left panel">
      <header>
        <div class="brand-lockup">
          <div class="brand-mark">S</div>
          <div>
            <h1>生成过程</h1>
            <div class="sub">LLM/VLM 推理与工具反馈实时流</div>
          </div>
        </div>
        <div class="status-row">
          <span id="genTimer" class="timer-pill">⏱ 00:00</span>
          <label id="isaacLine" class="switch-line"><span class="switch-text">Isaac Sim</span>
            <span class="switch"><input id="isaacToggle" type="checkbox"><span class="slider"></span></span>
            <span id="isaacSpinner" class="spinner"></span>
            <span id="isaacStateText" class="sub">未运行</span>
          </label>
          <label id="trellisLine" class="switch-line"><span class="switch-text">Trellis</span>
            <span class="switch"><input id="trellisToggle" type="checkbox"><span class="slider"></span></span>
            <span id="trellisSpinner" class="spinner"></span>
            <span id="trellisStateText" class="sub">未运行</span>
          </label>
        </div>
      </header>

      <div id="chat" class="chat"></div>

      <div class="composer">
        <div class="task-opts">
          <label class="opt-label">机器人
            <select id="robotType">
              <option value="">无（纯场景生成）</option>
              <option value="franka">Franka 机械臂（固定，抓放）</option>
              <option value="mobile franka">移动 Franka（导航 + 抓放）</option>
              <option value="unitree g1">宇树 G1 人形（仅导航）</option>
            </select>
          </label>
          <label class="opt-label" id="roomTypeWrap" style="display:none;">房间类型
            <input id="roomType" type="text" placeholder="living room" />
          </label>
        </div>
        <textarea id="prompt" placeholder="描述你想生成的 3D 场景，例如：暖色现代餐厅，橡木桌椅，绿植，开放式置物架，桌面有餐具与细节装饰。"></textarea>
        <div class="controls">
          <div id="runHint" class="hint">Isaac Sim MCP 服务器运行后才能开始生成。</div>
          <div>
            <button id="clearBtn" class="secondary">清空对话</button>
            <button id="stopBtn" class="danger" disabled>终止</button>
            <button id="generateBtn" disabled>发送生成</button>
          </div>
        </div>
      </div>
    </section>

    <section class="right panel">
      <header>
        <div class="brand-lockup">
          <div>
            <h2>生成结果</h2>
            <div class="sub">场景渲染与布局概要</div>
          </div>
        </div>
        <span id="layoutChip" class="layout-chip" title="点击复制 Layout ID">⬡ <span id="layoutChipId"></span></span>
      </header>

      <div class="results">
        <div id="pbarWrap" class="pbar-wrap">
          <div class="pbar-head">
            <span class="pcount" id="pbarCount">工具调用 0/0</span>
            <span class="peta" id="pbarEta"></span>
          </div>
          <div class="pbar"><div id="pbarFill" class="pbar-fill"></div></div>
        </div>

        <div class="progress" id="progress">
          <div class="pstep" data-step="input"><div class="pdot">1</div><div class="plabel">解析需求</div></div>
          <div class="pstep" data-step="layout"><div class="pdot">2</div><div class="plabel">房间布局</div></div>
          <div class="pstep" data-step="objects"><div class="pdot">3</div><div class="plabel">物体摆放</div></div>
          <div class="pstep" data-step="export"><div class="pdot">4</div><div class="plabel">导出资产</div></div>
        </div>

        <div id="resultsEmpty" class="results-empty">
          <div class="big">🏠</div>
          <div>
            <div style="font-weight:680;color:var(--text);">还没有生成结果</div>
            <div class="sub" style="margin-top:4px;">在左侧输入场景描述并点击「发送生成」，<br/>渲染图与布局概要会在这里逐步出现。</div>
          </div>
        </div>

        <div id="heroWrap" class="hero-wrap">
          <div class="section-label">场景渲染</div>
          <img id="hero" class="hero" alt="scene render" />
          <div id="heroCap" class="hero-cap"></div>
          <div id="thumbs" class="thumbs"></div>
        </div>

        <div id="summary" class="summary">
          <div class="section-label">布局概要</div>
          <div class="summary-grid">
            <div class="stat"><div class="k">房间类型</div><div class="v" id="statRoom">—</div></div>
            <div class="stat"><div class="k">物体数量</div><div class="v" id="statObjects">—</div></div>
            <div class="stat"><div class="k">房间尺寸</div><div class="v" id="statSize">—</div></div>
            <div class="stat"><div class="k">USD 资产</div><div class="v" id="statAssets">—</div></div>
          </div>
        </div>
      </div>
    </section>

    <section class="logbar panel">
      <div class="panel-title">
        <span>命令行 log</span>
        <span class="sub">后台生成过程的实时输出</span>
      </div>
      <div id="termLog" class="term"><span class="term-empty">等待后台输出…</span></div>
    </section>
  </main>

  <div id="lightbox" class="lightbox"><img id="lightboxImg" alt="preview" /></div>

  <script>
    const chat = document.getElementById('chat');
    const promptEl = document.getElementById('prompt');
    const generateBtn = document.getElementById('generateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const stopBtn = document.getElementById('stopBtn');
    const isaacToggle = document.getElementById('isaacToggle');
    const trellisToggle = document.getElementById('trellisToggle');
    const isaacLine = document.getElementById('isaacLine');
    const trellisLine = document.getElementById('trellisLine');
    const isaacStateText = document.getElementById('isaacStateText');
    const trellisStateText = document.getElementById('trellisStateText');
    const runHint = document.getElementById('runHint');
    const genTimer = document.getElementById('genTimer');
    const termLog = document.getElementById('termLog');
    const pendingServices = { isaac: null, trellis: null };
    const pendingSince = { isaac: 0, trellis: 0 };
    let keepChatAtBottom = true;
    let keepTermAtBottom = true;
    let serviceStatus = { isaac: false, trellis: false, generating: false };
    let wasGenerating = false;
    let timerStart = 0;
    let timerInterval = null;
    const stages = ['input', 'layout', 'objects', 'export'];
    const shownImageUrls = new Set();

    // ---- right results panel elements ----
    const resultsEmpty = document.getElementById('resultsEmpty');
    const heroWrap = document.getElementById('heroWrap');
    const heroImg = document.getElementById('hero');
    const heroCap = document.getElementById('heroCap');
    const thumbs = document.getElementById('thumbs');
    const summaryCard = document.getElementById('summary');
    const layoutChip = document.getElementById('layoutChip');
    const layoutChipId = document.getElementById('layoutChipId');
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightboxImg');
    const resultsPane = document.querySelector('.results');
    const pbarWrap = document.getElementById('pbarWrap');
    const pbarFill = document.getElementById('pbarFill');
    const pbarCount = document.getElementById('pbarCount');
    const pbarEta = document.getElementById('pbarEta');
    let keepResultsAtBottom = true;
    let heroFollowLatest = true;

    function revealResults() { resultsEmpty.classList.add('hide'); }

    function isNearBottom(el, threshold = 50) {
      return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
    }

    function stickResults() {
      if (keepResultsAtBottom) {
        requestAnimationFrame(() => { resultsPane.scrollTop = resultsPane.scrollHeight; });
      }
    }
    resultsPane.addEventListener('scroll', () => {
      keepResultsAtBottom = isNearBottom(resultsPane, 80);
    });

    // ---- generation timer (start -> running -> stops when finished) ----
    function formatElapsed(ms) {
      const total = Math.max(0, Math.floor(ms / 1000));
      const mm = String(Math.floor(total / 60)).padStart(2, '0');
      const ss = String(total % 60).padStart(2, '0');
      return `${mm}:${ss}`;
    }
    function startTimer() {
      if (timerInterval) return;
      timerStart = Date.now();
      genTimer.classList.remove('done');
      genTimer.classList.add('show', 'running');
      genTimer.textContent = '⏱ 00:00';
      timerInterval = setInterval(() => {
        genTimer.textContent = '⏱ ' + formatElapsed(Date.now() - timerStart);
      }, 250);
    }
    function stopTimer(suffix) {
      if (!timerInterval) return;
      clearInterval(timerInterval);
      timerInterval = null;
      genTimer.classList.remove('running');
      genTimer.classList.add('done');
      genTimer.textContent = '⏱ ' + formatElapsed(Date.now() - timerStart) + ' · ' + (suffix || '已完成');
    }

    // ---- tool-call progress bar + ETA ----
    function resetProgress() {
      pbarWrap.classList.remove('show');
      pbarFill.classList.remove('done');
      pbarFill.style.width = '0%';
      pbarCount.textContent = '工具调用 0/0';
      pbarEta.textContent = '';
    }
    function updateProgress(current, total, done) {
      if (!total || total <= 0) return;
      pbarWrap.classList.add('show');
      const pct = Math.max(0, Math.min(100, (current / total) * 100));
      pbarFill.style.width = pct.toFixed(1) + '%';
      pbarCount.textContent = `工具调用 ${current}/${total}`;
      if (done || current >= total) {
        pbarFill.classList.add('done');
        pbarEta.textContent = '已完成';
        return;
      }
      pbarFill.classList.remove('done');
      // Estimate remaining time from the average time per tool call so far.
      if (current >= 1 && timerStart) {
        const elapsed = Date.now() - timerStart;
        const remainMs = (elapsed / current) * (total - current);
        pbarEta.textContent = '预计剩余 ' + formatElapsed(remainMs);
      } else {
        pbarEta.textContent = '预估中…';
      }
    }

    // ---- raw backend log (bottom terminal) ----
    function appendTermLog(message, time) {
      if (!message) return;
      const empty = termLog.querySelector('.term-empty');
      if (empty) empty.remove();
      const stick = keepTermAtBottom || isNearBottom(termLog, 80);
      message.split('\n').forEach(part => {
        const line = document.createElement('div');
        line.className = 'term-line';
        if (/error|traceback|failed|exception|退出码 [^0]/i.test(part)) line.classList.add('err');
        line.textContent = (time ? `[${time}] ` : '') + part;
        termLog.appendChild(line);
      });
      while (termLog.childElementCount > 2000) termLog.removeChild(termLog.firstChild);
      if (stick) {
        requestAnimationFrame(() => { termLog.scrollTop = termLog.scrollHeight; });
      }
    }
    termLog.addEventListener('scroll', () => {
      keepTermAtBottom = isNearBottom(termLog, 80);
    });

    function setStage(activeStage, completed = []) {
      document.querySelectorAll('#progress .pstep').forEach(step => {
        const name = step.dataset.step;
        step.classList.toggle('active', name === activeStage);
        step.classList.toggle('done', completed.includes(name));
      });
    }

    function setHero(url, caption) {
      heroImg.src = url;
      heroCap.textContent = caption || '';
      heroWrap.classList.add('show');
      revealResults();
      thumbs.querySelectorAll('.thumb').forEach(t => t.classList.toggle('active', t.dataset.url === url));
      stickResults();
    }

    function addSceneImage(url, caption) {
      if (!url || shownImageUrls.has(url)) return;
      shownImageUrls.add(url);
      const t = document.createElement('img');
      t.className = 'thumb';
      t.src = url;
      t.alt = caption || 'render';
      t.dataset.url = url;
      t.title = caption || '';
      // Clicking a thumbnail pins it and pauses auto-follow so you can inspect it.
      t.addEventListener('click', () => { heroFollowLatest = false; setHero(url, caption); });
      // Newest thumbnail goes first so the latest render is easy to find.
      thumbs.insertBefore(t, thumbs.firstChild);
      // The big hero preview always tracks the latest generated image (unless the
      // user has clicked an older thumbnail to inspect it).
      if (heroFollowLatest || !heroImg.src) setHero(url, caption);
      stickResults();
    }

    heroImg.addEventListener('click', () => {
      if (!heroImg.src) return;
      lightboxImg.src = heroImg.src;
      lightbox.classList.add('show');
    });
    lightbox.addEventListener('click', () => lightbox.classList.remove('show'));
    layoutChip.addEventListener('click', () => {
      const id = layoutChipId.textContent.trim();
      if (id) navigator.clipboard?.writeText(id);
    });

    function updateStagesFromLog(line) {
      const text = line || '';
      if (/系统发给\s*LLM\/VLM|开始生成|Task: Generate/i.test(text)) {
        setStage('input', []);
      } else if (/generate_room_layout|Room structure|layout|floor plan|房间|布局/i.test(text)) {
        setStage('layout', ['input']);
      } else if (/object|furniture|place|Trellis|物体|家具|放置/i.test(text)) {
        setStage('objects', ['input', 'layout']);
      } else if (/usd|usdz|export|save|Layout ID|生成完成|导出/i.test(text)) {
        setStage('export', ['input', 'layout', 'objects']);
      }
    }

    function addMessage(type, title, body, time) {
      const shouldStick = keepChatAtBottom || isNearBottom(chat);
      const div = document.createElement('div');
      div.className = `msg ${type}`;
      const when = time || new Date().toLocaleTimeString();
      div.innerHTML = `<div class="msg-head"><span>${title}</span><span>${when}</span></div><div class="msg-body"></div>`;
      div.querySelector('.msg-body').textContent = body;
      chat.appendChild(div);
      if (shouldStick) {
        requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
      }
    }

    function addImageMessage(title, imageUrl, caption, time) {
      if (!imageUrl || shownImageUrls.has(imageUrl)) return;
      shownImageUrls.add(imageUrl);
      const shouldStick = keepChatAtBottom || isNearBottom(chat);
      const div = document.createElement('div');
      div.className = 'msg image';
      const when = time || new Date().toLocaleTimeString();
      div.innerHTML = `<div class="msg-head"><span>${title}</span><span>${when}</span></div><div class="msg-body"></div>`;
      const body = div.querySelector('.msg-body');
      const img = document.createElement('img');
      img.className = 'chat-image';
      img.src = imageUrl;
      img.alt = caption || 'image';
      body.appendChild(img);
      if (caption) {
        const cap = document.createElement('div');
        cap.className = 'image-caption';
        cap.textContent = caption;
        body.appendChild(cap);
      }
      chat.appendChild(div);
      if (shouldStick) {
        requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
      }
    }

    chat.addEventListener('scroll', () => {
      keepChatAtBottom = isNearBottom(chat, 80);
    });

    function chatEventFromLog(line) {
      const text = (line || '').trim();
      if (!text) return null;

      if (isNoiseLog(text)) return null;

      if (/error|traceback|failed|exception|生成失败/i.test(text)) {
        return ['error', '错误', text];
      }

      if (/^执行命令：/.test(text)) return null;
      if (/^\[(isaac|trellis)\]/i.test(text)) {
        if (/进程结束|退出码/i.test(text)) return ['server', '服务状态', text];
        return null;
      }

      // The verbose task instruction is noise in the chat; the concise room
      // description is already shown as the user's message.
      if (/系统发给\s*LLM\/VLM|发给\s*LLM|发给\s*VLM/i.test(text)) {
        return null;
      }

      if (/🔤 Response:/i.test(text)) {
        return ['model', 'LLM/VLM 反馈', tidy(text.replace(/^.*?🔤 Response:\s*/i, ''))];
      }

      if (/Qwen|Claude|OpenAI|LLM|VLM|assistant|thinking|model response|模型反馈|视觉模型|语言模型/i.test(text)) {
        return ['model', 'LLM/VLM 反馈', tidy(text)];
      }

      if (/^(Semantic critic|Physics critic|语义评估|物理评估)/i.test(text)) {
        return ['tool', '评估反馈', text];
      }

      if (/Reached maximum tool call limit/i.test(text)) {
        return ['error', '生成中止', '达到工具调用上限，生成流程已停止。'];
      }

      if (/^开始生成|机器人任务流程已跳过|生成进程结束|生成完成/.test(text)) {
        return ['server', '服务端返回', text];
      }

      if (/Layout ID:\s*layout_|layout_[0-9a-fA-F]{8}/.test(text)) {
        return ['server', '生成结果', text];
      }

      return null;
    }

    // Keep chat messages short and readable: collapse blank runs and clip very
    // long blobs so parameter dumps don't bloat a bubble.
    function tidy(text) {
      let out = (text || '').replace(/\n{3,}/g, '\n\n').trim();
      if (out.length > 600) out = out.slice(0, 600).trimEnd() + ' …';
      return out;
    }

    function isNoiseLog(text) {
      return [
        // paths, directories and CLI parameters are not useful in the chat
        /^\s*\//,
        /^\s*\.\.?\//,
        /^[\w.\-]+\/[^\s]*$/,
        /(^|\s)--[a-z][\w-]+/i,
        /\b(server_paths|max_tool_calls|room_desc|generation_python|cwd)\b\s*[:=]/i,
        /(保存到|已保存到?|写入到?|输出目录|工作目录|written to|output (?:dir|directory|path))/i,
        /^(目录|路径|输出目录|工作目录|文件)[:：]/,
        /Client chat log saved to:/i,
        /Enhanced chat log saved to:/i,
        /^Chat log saved to:/i,
        /^💾 Chat log saved to:/i,
        /^💾 Enhanced chat log saved to:/i,
        /^Auto-saving chat log/i,
        /^Saved chat log/i,
        /^📊 Token usage:/i,
        /^🔧 Tool usage:/i,
        /^💬 Conversation summary:/i,
        /^👤 User messages:/i,
        /^🤖 Assistant messages:/i,
        /^🧠 Reasoning entries:/i,
        /^🔧 Tool calls:/i,
        /^📤 Tool results:/i,
        /^📝 Content length:/i,
        /^🤖 Qwen3-VL Response:/i,
        /^🔧 Executing \d+ tool call/i,
        /^🔧 Found \d+ tool call/i,
        /^🔧 Parsed \d+ tool call/i,
        /^🔧 Tool \d+\/\d+:/i,
        /^📥 Arguments:/i,
        /^📤 Result:/i,
        /^Tool \d+\/\d+:/i,
        /^Arguments:/i,
        /^Result:/i,
        /^[{}\[\],:;\s]+$/,
        /^📊 Tool call count:/i,
        /^🔄 Continuing to next iteration/i,
        /^Loading image:/i,
        /^📸 Loading image:/i,
        /^✅ Image loaded:/i,
        /^Found \d+ image\(s\) to include/i,
        /^📸 Found \d+ image\(s\) to include/i,
        /^Including \d+ image\(s\) in the query/i,
        /^📸 Including \d+ image\(s\) in the query/i,
        /^执行命令：/i,
        /^\[trellis\].*(Local:|Network:|Docs Page|OpenAPI|Health Check|Metrics|Share the Network URL|firewall|API Endpoints)/i,
        /^\[isaac\].*(INFO|DEBUG|omni\.|carb\.|kit)/i
      ].some(pattern => pattern.test(text));
    }

    function normalizedService(service) {
      if (typeof service === 'boolean') return { running: service, status: service ? 'running' : 'stopped' };
      return service || { running: false, status: 'stopped' };
    }

    function serviceLabel(service) {
      if (service.running) return '运行中';
      if (service.status === 'starting') return '启动中/未就绪';
      return '未运行';
    }

    function setPill(el, service, name) {
      service = normalizedService(service);
      const tone = service.running ? 'ok' : service.status === 'starting' ? 'warn' : 'bad';
      el.className = `pill ${tone}`;
      el.textContent = `${name} ${serviceLabel(service)}`;
      if (service.message) el.title = service.message;
    }

    function updateServiceControl(name, service) {
      service = normalizedService(service);
      const line = name === 'isaac' ? isaacLine : trellisLine;
      const toggle = name === 'isaac' ? isaacToggle : trellisToggle;
      const text = name === 'isaac' ? isaacStateText : trellisStateText;
      const running = Boolean(service.running);
      const starting = service.status === 'starting';

      if ((pendingServices[name] === 'start' && running) || (pendingServices[name] === 'stop' && !running)) {
        pendingServices[name] = null;
      }
      if (pendingServices[name] === 'start' && service.status === 'stopped' && Date.now() - pendingSince[name] > 2000) {
        pendingServices[name] = null;
      }
      if (pendingServices[name] && Date.now() - pendingSince[name] > 30000) {
        pendingServices[name] = null;
      }

      const pending = pendingServices[name];
      line.classList.toggle('loading', Boolean(pending) || starting);
      toggle.disabled = Boolean(pending);
      toggle.checked = pending === 'start' ? true : pending === 'stop' ? false : running || starting;
      text.textContent = pending === 'start'
        ? '正在打开'
        : pending === 'stop'
          ? '正在关闭'
          : serviceLabel(service);
      if (service.message) text.title = service.message;
    }

    async function refreshStatus() {
      const res = await fetch('/status.json');
      serviceStatus = await res.json();
      const generating = Boolean(serviceStatus.generation && serviceStatus.generation.running);
      if (generating && !timerInterval) startTimer();
      if (!generating && wasGenerating) stopTimer();
      wasGenerating = generating;
      updateServiceControl('isaac', serviceStatus.isaac);
      updateServiceControl('trellis', serviceStatus.trellis);
      // Isaac no longer gates the button: /api/generate auto-starts the headless
      // kit (and closes any GUI display) before running, waiting for the port.
      generateBtn.disabled = serviceStatus.generation.running;
      stopBtn.disabled = !serviceStatus.generation.running;
      runHint.textContent = serviceStatus.generation.running
        ? '生成正在进行中。'
        : (serviceStatus.isaac.running
          ? '可以开始生成。'
          : 'Isaac Sim 未运行——点「发送生成」会自动以无头模式拉起（首次需等待数分钟）。');
    }

    async function toggleService(name, checked) {
      const action = checked ? 'start' : 'stop';
      pendingServices[name] = action;
      pendingSince[name] = Date.now();
      updateServiceControl(name, checked);
      addMessage('local', '本地发送', `${name} ${action}`);
      const res = await fetch('/service', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, action })
      });
      const data = await res.json();
      if (!data.ok) pendingServices[name] = null;
      addMessage(data.ok ? 'server' : 'error', '服务端返回', data.message);
      await refreshStatus();
    }

    isaacToggle.addEventListener('change', e => toggleService('isaac', e.target.checked));
    trellisToggle.addEventListener('change', e => toggleService('trellis', e.target.checked));
    function resetResults() {
      shownImageUrls.clear();
      heroFollowLatest = true;
      resetProgress();
      thumbs.innerHTML = '';
      heroImg.removeAttribute('src');
      heroCap.textContent = '';
      heroWrap.classList.remove('show');
      summaryCard.classList.remove('show');
      layoutChip.classList.remove('show');
      resultsEmpty.classList.remove('hide');
      ['statRoom','statObjects','statSize','statAssets'].forEach(id => { document.getElementById(id).textContent = '—'; });
      setStage('input', []);
    }

    clearBtn.addEventListener('click', () => {
      chat.innerHTML = '';
      termLog.innerHTML = '<span class="term-empty">等待后台输出…</span>';
      resetResults();
    });

    const robotTypeEl = document.getElementById('robotType');
    const roomTypeEl = document.getElementById('roomType');
    const roomTypeWrap = document.getElementById('roomTypeWrap');
    const SCENE_PLACEHOLDER = '描述你想生成的 3D 场景，例如：暖色现代餐厅，橡木桌椅，绿植，开放式置物架，桌面有餐具与细节装饰。';
    const ROBOT_PLACEHOLDER = '描述机器人任务，例如：In a living room with a coffee table holding a toy cube and a plate, the robot must pick up the cube and place it on the plate.（G1 仅支持导航任务：walk to ...）';
    robotTypeEl.addEventListener('change', () => {
      const robotMode = !!robotTypeEl.value;
      roomTypeWrap.style.display = robotMode ? '' : 'none';
      promptEl.placeholder = robotMode ? ROBOT_PLACEHOLDER : SCENE_PLACEHOLDER;
    });

    generateBtn.addEventListener('click', async () => {
      const roomDesc = promptEl.value.trim();
      const robotType = robotTypeEl.value;
      if (!roomDesc) {
        addMessage('error', '输入为空', robotType ? '请输入机器人任务描述。' : '请输入想要生成的房间描述。');
        return;
      }
      promptEl.value = '';
      resetResults();
      // Default to showing the latest content in both panels.
      keepChatAtBottom = true;
      keepResultsAtBottom = true;
      keepTermAtBottom = true;
      startTimer();
      const label = robotType ? `[机器人任务 · ${robotType}] ` : '';
      addMessage('local', '本地发送', label + roomDesc);
      const res = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room_desc: roomDesc,
          robot_type: robotType,
          room_type: (roomTypeEl.value || '').trim(),
        })
      });
      const data = await res.json();
      addMessage(data.ok ? 'server' : 'error', '服务端返回', data.message);
      await refreshStatus();
    });

    stopBtn.addEventListener('click', async () => {
      stopBtn.disabled = true;
      // Stop the timer + clear the current task immediately for instant feedback.
      stopTimer('已终止');
      addMessage('local', '本地发送', '终止生成');
      try {
        const res = await fetch('/generate/stop', { method: 'POST' });
        const data = await res.json();
        addMessage(data.ok ? 'server' : 'error', '服务端返回', data.message);
      } catch (e) {
        addMessage('error', '终止失败', String(e));
      }
      // Clear the task output (results panel + progress); keep chat/terminal history.
      resetResults();
      await refreshStatus();
    });

    function handleLogEvent(event) {
      const item = JSON.parse(event.data);
      appendTermLog(item.message || '', item.time);
      updateStagesFromLog(item.message || '');
      const chatEvent = chatEventFromLog(item.message || '');
      if (!chatEvent) return;
      const [type, title, body] = chatEvent;
      addMessage(type, title, body, item.time);
    }

    function handleResultEvent(event) {
      const item = JSON.parse(event.data);
      stopTimer();
      addMessage('server', '生成完成', `Layout ID: ${item.layout_id}`);
      setStage(null, stages);
      revealResults();
      if (item.layout_id) {
        layoutChipId.textContent = item.layout_id;
        layoutChip.classList.add('show');
        fetchSummary(item.layout_id, item.summary);
      }
    }

    function handleImageEvent(event) {
      const item = JSON.parse(event.data);
      // Scene renders go to the right results gallery; other images too.
      addSceneImage(item.url, item.caption || item.name || item.title || '');
    }

    function handleProgressEvent(event) {
      const item = JSON.parse(event.data);
      updateProgress(item.current, item.total, item.done);
    }

    function dispatchSseEvent(type, data) {
      const event = { data: data || '{}' };
      if (type === 'log') handleLogEvent(event);
      else if (type === 'result') handleResultEvent(event);
      else if (type === 'image') handleImageEvent(event);
      else if (type === 'progress') handleProgressEvent(event);
      else if (type === 'status') refreshStatus();
    }

    async function connectEventsWithFetch(lastEventId = '') {
      let cursor = lastEventId;
      while (true) {
        try {
          const headers = cursor ? { 'Last-Event-ID': cursor } : {};
          const res = await fetch('/events', { headers, cache: 'no-store' });
          if (!res.ok || !res.body) throw new Error(`SSE HTTP ${res.status}`);
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const chunks = buffer.split(/\n\n/);
            buffer = chunks.pop() || '';
            for (const chunk of chunks) {
              let type = 'message';
              let data = '';
              for (const line of chunk.split(/\n/)) {
                if (line.startsWith('id:')) cursor = line.slice(3).trim();
                else if (line.startsWith('event:')) type = line.slice(6).trim();
                else if (line.startsWith('data:')) data += line.slice(5).trimStart();
              }
              if (data) dispatchSseEvent(type, data);
            }
          }
        } catch (e) {
          appendTermLog(`事件流断开，正在重连：${e.message || e}`, new Date().toLocaleTimeString());
        }
        await new Promise(resolve => setTimeout(resolve, 1500));
      }
    }

    function connectEvents() {
      if (typeof EventSource !== 'undefined') {
        const events = new EventSource('/events');
        events.addEventListener('log', handleLogEvent);
        events.addEventListener('result', handleResultEvent);
        events.addEventListener('image', handleImageEvent);
        events.addEventListener('progress', handleProgressEvent);
        events.addEventListener('status', () => refreshStatus());
        events.onerror = () => appendTermLog('事件流连接异常，浏览器会自动重连。', new Date().toLocaleTimeString());
        return;
      }
      connectEventsWithFetch();
    }

    function fillSummary(s) {
      if (!s) return;
      if (s.room_type) document.getElementById('statRoom').textContent = s.room_type;
      if (s.object_count != null) document.getElementById('statObjects').innerHTML = `${s.object_count} <small>件</small>`;
      if (s.room_size) document.getElementById('statSize').innerHTML = `${s.room_size} <small>m</small>`;
      if (s.asset_count != null) document.getElementById('statAssets').innerHTML = `${s.asset_count} <small>个</small>`;
      summaryCard.classList.add('show');
      stickResults();
    }

    async function fetchSummary(layoutId, inlineSummary) {
      if (inlineSummary) { fillSummary(inlineSummary); return; }
      try {
        const res = await fetch('/layout-json/' + encodeURIComponent(layoutId));
        if (res.ok) fillSummary(await res.json());
      } catch (e) { /* summary is best-effort */ }
    }

    connectEvents();
    refreshStatus();
    setInterval(refreshStatus, 3000);
  </script>
</body>
</html>
"""


class ManagedProcess:
    def __init__(self, name: str, command: str, cwd: Path):
        self.name = name
        self.command = command
        self.cwd = cwd
        self.process: subprocess.Popen[str] | None = None

    def running(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        return process_exists(self.name)

    def start(self, emit) -> str:
        if self.process and self.process.poll() is None:
            return f"{self.name} 已在当前控制台运行。"
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        threading.Thread(target=stream_process, args=(self.name, self.process, emit), daemon=True).start()
        return f"{self.name} 启动中：{self.command}"

    def stop(self) -> str:
        if self.process and self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            return f"{self.name} 已发送停止信号。"
        if process_exists(self.name):
            try:
                subprocess.run(["pkill", "-f", self._process_pattern()], check=False)
                return f"{self.name} 已发送停止信号（外部进程）。"
            except Exception:
                return f"无法停止 {self.name} 外部进程。"
        return f"{self.name} 不在运行中，无需停止。"

    def _process_pattern(self) -> str:
        patterns = {"isaac": "isaac.sim.mcp_extension", "trellis": "trellis_flask_server.py"}
        return patterns.get(self.name, self.name)


def process_exists(name: str) -> bool:
    patterns = {
        "isaac": "isaac.sim.mcp_extension",
        "trellis": "trellis_flask_server.py",
    }
    pattern = patterns.get(name, name)
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        current_pid = str(os.getpid())
        return any(pid and pid != current_pid for pid in result.stdout.splitlines())
    except Exception:
        return False


def slurm_job_id_to_port(job_id: str | None, port_start: int = 8080, port_end: int = 40000) -> int:
    job_id_str = str(job_id)
    hash_int = int(hashlib.md5(job_id_str.encode()).hexdigest(), 16)
    return port_start + (hash_int % (port_end - port_start + 1))


def isaac_mcp_port() -> int:
    override = os.environ.get("SAGE_ISAAC_MCP_PORT") or os.environ.get("ISAAC_MCP_PORT")
    if override:
        return int(override)
    return slurm_job_id_to_port(os.environ.get("SLURM_JOB_ID"))


def tcp_port_listening(port: int) -> bool:
    port_hex = f"{port:04X}"
    for proc_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = proc_path.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            local_address, state = parts[1], parts[3]
            if state == "0A" and local_address.rsplit(":", 1)[-1].upper() == port_hex:
                return True
    return False


def probe_isaac_mcp(timeout: float = 2.0) -> tuple[bool, str]:
    port = isaac_mcp_port()
    if tcp_port_listening(port):
        return True, f"MCP 端口 {port} 已开始监听。"
    return False, f"MCP 端口 {port} 尚未监听。"


def isaac_service_status(proc: ManagedProcess) -> dict[str, Any]:
    process_running = proc.running()
    if not process_running:
        return {
            "running": False,
            "process_running": False,
            "status": "stopped",
            "port": isaac_mcp_port(),
            "message": "未检测到 Isaac Sim 进程。",
        }
    ready, message = probe_isaac_mcp()
    return {
        "running": ready,
        "process_running": process_running,
        "status": "running" if ready else "starting",
        "port": isaac_mcp_port(),
        "message": message,
    }


def process_service_status(proc: ManagedProcess) -> dict[str, Any]:
    running = proc.running()
    return {
        "running": running,
        "process_running": running,
        "status": "running" if running else "stopped",
        "message": f"{proc.name} 进程{'正在运行' if running else '未运行'}。",
    }


def stream_process(name: str, process: subprocess.Popen[str], emit) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        message = f"[{name}] {line.rstrip()}"
        emit("log", {"message": message})
        emit_images_from_line(message, emit)
    emit("log", {"message": f"[{name}] 进程结束，退出码 {process.poll()}。"})
    emit("status", {})


class AppState:
    def __init__(self, isaac_cmd: str, trellis_cmd: str, generation_python: str, max_tool_calls: int):
        self.events: list[tuple[int, str, dict[str, Any]]] = []
        self.next_event_id = 1
        self.cond = threading.Condition()
        self.generation_process: subprocess.Popen[str] | None = None
        self.generation_python = generation_python
        self.max_tool_calls = max_tool_calls
        self.latest_layout_id: str | None = None
        self.seen_image_keys: set[str] = set()
        self.isaac = ManagedProcess("isaac", isaac_cmd, SAGE_ROOT)
        self.trellis = ManagedProcess("trellis", trellis_cmd, SAGE_ROOT)
        # Post-generation GUI display process (scene viewer or robot-task run).
        self.display_process: subprocess.Popen[str] | None = None
        # Browser-connection tracking for idle auto-shutdown.
        self.sse_lock = threading.Lock()
        self.sse_clients = 0
        self.idle_timer: threading.Timer | None = None
        self.shutdown_server = None  # set in main(): callable that stops the HTTP server
        self._shutdown_done = False
        try:
            self.idle_shutdown_seconds = int(os.environ.get("SAGE_WEB_IDLE_SHUTDOWN_SECONDS", "15"))
        except ValueError:
            self.idle_shutdown_seconds = 15

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("time", time.strftime("%H:%M:%S"))
        with self.cond:
            self.events.append((self.next_event_id, event_type, payload))
            self.next_event_id += 1
            self.events = self.events[-800:]
            self.cond.notify_all()

    def generation_running(self) -> bool:
        return self.generation_process is not None and self.generation_process.poll() is None

    def stop_display(self) -> None:
        """Close the post-generation GUI display process (viewer / task run)."""
        proc = self.display_process
        self.display_process = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def stop_generation(self) -> str:
        """Terminate the running generation (client + its MCP layout server) WITHOUT
        touching the Isaac Sim or Trellis services."""
        proc = self.generation_process
        killed = False
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                killed = True
            except Exception:
                try:
                    proc.terminate()
                    killed = True
                except Exception:
                    pass
        # Clean up any stray generation children. These patterns are generation-only
        # and never match the Isaac (isaac.sim.mcp_extension) or Trellis processes.
        # ("server/layout.py" is anchored with the dir prefix so the regex cannot
        # accidentally match other *layout*.py processes.)
        for pattern in ("client_generation_room_desc.py", "layout_wo_robot.py",
                        "client_generation_robot_task.py", "server/layout.py"):
            try:
                subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
            except Exception:
                pass
        self.generation_process = None
        return "已发送终止信号，生成任务已停止（Isaac / Trellis 未受影响）。" if killed \
            else "当前没有正在运行的生成任务（已顺带清理可能的残留进程）。"

    def shutdown_all(self) -> None:
        """Terminate EVERYTHING this console started: generation + Isaac + Trellis.
        Idempotent. Used on console exit and on browser-idle auto-shutdown."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        print("🧹 关闭控制台：正在停止生成任务、GUI 展示、Isaac Sim、Trellis…", file=sys.stderr)
        try:
            self.stop_display()
        except Exception as exc:
            print(f"stop_display failed: {exc}", file=sys.stderr)
        try:
            print(self.stop_generation(), file=sys.stderr)
        except Exception as exc:
            print(f"stop_generation failed: {exc}", file=sys.stderr)
        for proc in (self.isaac, self.trellis):
            try:
                print(proc.stop(), file=sys.stderr)
            except Exception as exc:
                print(f"{proc.name} stop failed: {exc}", file=sys.stderr)

    def client_connected(self) -> None:
        with self.sse_lock:
            self.sse_clients += 1
            if self.idle_timer is not None:
                self.idle_timer.cancel()
                self.idle_timer = None

    def client_disconnected(self) -> None:
        with self.sse_lock:
            self.sse_clients = max(0, self.sse_clients - 1)
            if self.sse_clients == 0 and self.idle_shutdown_seconds > 0:
                if self.idle_timer is not None:
                    self.idle_timer.cancel()
                self.idle_timer = threading.Timer(self.idle_shutdown_seconds, self._idle_shutdown)
                self.idle_timer.daemon = True
                self.idle_timer.start()

    def _idle_shutdown(self) -> None:
        with self.sse_lock:
            if self.sse_clients > 0:
                return  # a browser reconnected within the grace period
        print(
            f"🚪 浏览器界面已关闭 {self.idle_shutdown_seconds}s 无重连，正在关停所有服务并退出控制台…",
            file=sys.stderr,
        )
        self.shutdown_all()
        if self.shutdown_server is not None:
            try:
                self.shutdown_server()
            except Exception as exc:
                print(f"server shutdown failed: {exc}", file=sys.stderr)


def latest_layout_result(layout_id: str) -> dict[str, Any]:
    layout_dir = RESULTS_ROOT / layout_id
    json_path = layout_dir / f"{layout_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(str(json_path))
    layout = json.loads(json_path.read_text())
    assets = []
    asset_paths: list[Path] = []
    asset_paths.extend(path for path in layout_dir.iterdir() if path.suffix.lower() in {".usd", ".usdz"})
    collection = layout_dir / f"{layout_id}_usd_collection"
    if collection.exists():
        asset_paths.extend(path for path in collection.rglob("*") if path.suffix.lower() in {".usd", ".usdz"})
    for path in sorted({path.resolve() for path in asset_paths}):
        try:
            rel = path.relative_to(SERVER_ROOT)
        except ValueError:
            continue
        assets.append({"name": path.name, "url": f"/{rel.as_posix()}"})
    return {
        "layout": layout,
        "json_path": str(json_path),
        "assets": assets,
    }


def layout_summary(layout_id: str) -> dict[str, Any] | None:
    """Compact summary of a generated layout for the results panel."""
    try:
        result = latest_layout_result(layout_id)
    except Exception:
        return None
    layout = result.get("layout", {})
    rooms = layout.get("rooms", []) or []
    object_count = sum(len(r.get("objects", []) or []) for r in rooms)
    room_type = "、".join(
        sorted({(r.get("room_type") or "").replace("_", " ") for r in rooms if r.get("room_type")})
    ) or "—"
    room_size = "—"
    if rooms:
        dim = rooms[0].get("dimensions", {}) or {}
        w, l = dim.get("width"), dim.get("length")
        if w and l:
            room_size = f"{float(w):.1f}×{float(l):.1f}"
    return {
        "layout_id": layout_id,
        "room_type": room_type,
        "object_count": object_count,
        "room_size": room_size,
        "asset_count": len(result.get("assets", []) or []),
    }


def find_layout_id(line: str) -> str | None:
    match = re.search(r"Layout ID:\s*(layout_[A-Za-z0-9_]+)", line)
    if match:
        return match.group(1)
    match = re.search(r"(layout_[0-9a-fA-F]{8})", line)
    return match.group(1) if match else None


def image_path_candidates(line: str) -> list[str]:
    candidates = []
    pattern = r"(?P<path>(?:/[^ \t\r\n,;'\")]+|(?:\.\.?/)?[^ \t\r\n,;'\")]+)\.(?:png|jpg|jpeg|webp|gif))"
    for match in re.finditer(pattern, line, flags=re.IGNORECASE):
        candidates.append(match.group("path").rstrip(".,;:"))
    return candidates


def resolve_local_path(path_text: str, cwd: Path | None = None, allowed_extensions: set[str] | None = None) -> Path | None:
    raw = unquote(path_text.strip().strip("'\""))
    candidates: list[Path] = []
    path = Path(raw)
    if path.is_absolute():
        candidates.append(path)
    else:
        bases = [cwd] if cwd else []
        bases.extend([CLIENT_ROOT, SERVER_ROOT, SAGE_ROOT])
        candidates.extend(base / raw for base in bases if base is not None)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        suffixes = allowed_extensions or IMAGE_EXTENSIONS
        if resolved.exists() and resolved.is_file() and resolved.suffix.lower() in suffixes:
            return resolved
    return None


def image_event_for_path(path: Path) -> dict[str, Any] | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not path_is_under(resolved, SAGE_ROOT):
        return None
    rel_name = resolved.name
    title = "Isaac 渲染图" if is_scene_render_image(resolved) else "图片"
    # Preview renders reuse fixed filenames and are overwritten every iteration.
    # Append the file mtime so an updated render is treated as a NEW image by the
    # browser (otherwise the identical URL is deduped and the panel never updates).
    try:
        version = int(resolved.stat().st_mtime)
    except OSError:
        version = 0
    return {
        "url": f"/image-file?path={quote(str(resolved))}&v={version}",
        "name": rel_name,
        "caption": rel_name,
        "title": title,
    }


def is_scene_render_image(path: Path) -> bool:
    name = path.name.lower()
    return any(
        token in name
        for token in ("rendered_view", "top_down_annotated", "room_", "wall_placement", "full_view")
    ) and "texture" not in name


def is_texture_image(path: Path) -> bool:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    return "texture" in name or "materials" in parts or "objaverse" in parts


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def emit_images_from_line(line: str, emit, cwd: Path | None = None) -> None:
    seen: set[Path] = set()
    for candidate in image_path_candidates(line):
        image_path = resolve_local_path(candidate, cwd)
        if image_path is None or image_path in seen:
            continue
        if is_texture_image(image_path) and not re.search(r"Loading image|Including .*image|用户图片|input image", line, re.I):
            continue
        seen.add(image_path)
        event = image_event_for_path(image_path)
        if event:
            emit("image", event)


def emit_images_from_chat_log_line(line: str, state: AppState) -> None:
    if "chat log saved to:" not in line.lower() or ".json" not in line.lower():
        return
    for candidate in re.findall(r"(?P<path>(?:/[^ \t\r\n]+|(?:\.\.?/)?[^ \t\r\n]+)\.json)", line):
        log_path = resolve_local_path(candidate, CLIENT_ROOT, {".json"})
        if log_path is None:
            continue
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for image in iter_chat_log_images(data):
            key = image.get("key") or image.get("url") or image.get("caption")
            if not key or key in state.seen_image_keys:
                continue
            state.seen_image_keys.add(key)
            state.emit("image", image)


def iter_chat_log_images(data: Any):
    if isinstance(data, dict):
        if data.get("type") == "image_url":
            image_url = (data.get("image_url") or {}).get("url")
            metadata = data.get("image_metadata") or {}
            if image_url:
                caption = metadata.get("filename") or metadata.get("path") or "输入图片"
                yield {
                    "url": image_url,
                    "name": metadata.get("filename") or "image",
                    "caption": caption,
                    "title": "用户图片",
                    "key": metadata.get("path") or image_url[:180],
                }
        for value in data.values():
            yield from iter_chat_log_images(value)
    elif isinstance(data, list):
        for item in data:
            yield from iter_chat_log_images(item)


def layout_room_ids(layout_id: str) -> list[str]:
    try:
        layout = latest_layout_result(layout_id)["layout"]
    except Exception:
        return []
    return [room.get("id") for room in layout.get("rooms", []) if room.get("id")]


def scene_render_images_for_layout(layout_id: str) -> list[Path]:
    room_ids = layout_room_ids(layout_id)
    if not room_ids:
        return []
    images: list[Path] = []
    preview_dir = RESULTS_ROOT / layout_id / "preview"
    if preview_dir.exists():
        # Whole-layout (all rooms in one frame) renders first — these show the
        # integrated building, not isolated per-room boxes.
        images.extend(preview_dir.glob(f"{layout_id}_full_view_*.png"))
        for room_id in room_ids:
            images.extend(preview_dir.glob(f"{room_id}_rendered_view_*.png"))

    vis_dir = SERVER_ROOT / "vis"
    if not vis_dir.exists():
        unique_images = {path.resolve() for path in images if path.is_file() and is_scene_render_image(path)}
        return sorted(unique_images, key=lambda path: (0, -path.stat().st_mtime))

    for room_id in room_ids:
        patterns = [
            f"{room_id}_rendered_view_*.png",
            f"{room_id}_top_down_annotated.png",
            f"room_{room_id}_*.png",
            f"wall_placement_{room_id}_*.png",
        ]
        for pattern in patterns:
            images.extend(vis_dir.glob(pattern))
    unique_images = {path.resolve() for path in images if path.is_file() and is_scene_render_image(path)}

    def image_priority(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        if "full_view" in name:
            rank = 0
        elif "rendered_view" in name:
            rank = 1
        elif "top_down_annotated" in name:
            rank = 2
        elif name.startswith("room_"):
            rank = 3
        else:
            rank = 4
        return (rank, -path.stat().st_mtime)

    return sorted(unique_images, key=image_priority)


def emit_scene_render_images(layout_id: str, state: AppState, limit: int = 6) -> None:
    for path in scene_render_images_for_layout(layout_id)[:limit]:
        # Key on path + mtime so a re-rendered (overwritten) preview re-emits.
        try:
            key = f"{path}:{int(path.stat().st_mtime)}"
        except OSError:
            key = str(path)
        if key in state.seen_image_keys:
            continue
        event = image_event_for_path(path)
        if not event:
            continue
        state.seen_image_keys.add(key)
        state.emit("image", event)


def room_generation_instruction(room_desc: str) -> str:
    # Display-only summary shown in the chat; the real task/prompt is built inside
    # client_generation_room_desc.py. The web flow now supports multi-room layouts.
    return (
        "Task: Generate a scene (one room, or a multi-room house/apartment if the "
        "description implies it) from the following description.\n\n"
        f"Description: {room_desc}\n\n"
        "Process:\n"
        "1. Generate the layout (single or multiple connected rooms as appropriate).\n"
        "2. For each room: place large furniture first.\n"
        "3. Add functional object groups and detailed surface objects.\n"
        "4. Add decorative/background objects; decorate every room.\n"
        "5. Do not run robot task generation."
    )


def wait_for_isaac_ready(state: AppState, timeout_seconds: int = 420) -> bool:
    """Block until the headless Isaac MCP kit is accepting connections."""
    deadline = time.time() + timeout_seconds
    notified = 0.0
    while time.time() < deadline:
        if isaac_service_status(state.isaac)["running"]:
            return True
        if time.time() - notified > 30:
            remaining = int(deadline - time.time())
            state.emit("log", {"message": f"等待 Isaac Sim（无头）就绪…（剩余 {remaining}s）"})
            notified = time.time()
        time.sleep(5)
    return isaac_service_status(state.isaac)["running"]


def launch_display_process(state: AppState, shell_cmd: str, name: str) -> None:
    """Spawn a GUI display process (scene viewer / robot-task run) and stream its
    output into the console log. The headless kit must already be stopped."""
    state.stop_display()
    state.emit("log", {"message": f"启动 GUI 展示（{name}）：{shell_cmd}"})
    process = subprocess.Popen(
        shell_cmd,
        cwd=str(SAGE_ROOT),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    state.display_process = process
    threading.Thread(target=stream_process, args=(f"display:{name}", process, state.emit), daemon=True).start()


def first_room_id(layout_id: str) -> str | None:
    try:
        layout = json.loads((RESULTS_ROOT / layout_id / f"{layout_id}.json").read_text())
        rooms = layout.get("rooms") or []
        return rooms[0].get("id") if rooms else None
    except Exception:
        return None


def launch_post_generation_display(state: AppState, layout_id: str, robot_type: str) -> None:
    """After a successful generation: stop the headless kit (frees VRAM for a GUI
    SimulationApp), then show the result — a lit scene viewer for plain scenes, or
    the robot-task runner for robot generations."""
    display_env = f"DISPLAY={_ISAAC_DISPLAY}"
    if not _display_available:
        state.emit("log", {"message": "未检测到 X display，跳过 GUI 展示（产物已在结果目录）。"})
        return

    state.emit("log", {"message": "生成完成 — 停止无头 Isaac kit，准备 GUI 展示…"})
    try:
        state.emit("log", {"message": state.isaac.stop()})
    except Exception as exc:
        state.emit("log", {"message": f"停止 Isaac kit 失败（继续尝试展示）：{exc}"})

    layout_dir = RESULTS_ROOT / layout_id
    if not robot_type:
        # Lit, ceiling-hidden viewer USD baked by render_layout_preview; fall back
        # to the plain combined USD (the viewer adds lights itself if missing).
        view_usd = layout_dir / f"{layout_id}_view.usd"
        if not view_usd.exists():
            view_usd = layout_dir / f"{layout_id}.usd"
        if not view_usd.exists():
            state.emit("log", {"message": f"未找到可展示的整体 USD（{layout_id}_view.usd / {layout_id}.usd），跳过 GUI 展示。"})
            return
        cmd = f"{display_env} {ISAAC_PYTHON_SH} {SCENE_VIEWER_SCRIPT} {view_usd}"
        launch_display_process(state, cmd, "场景查看器")
        return

    if robot_type == "unitree g1":
        nav_json = layout_dir / f"{layout_id}_g1_nav_path.json"
        if not nav_json.exists():
            state.emit("log", {"message": "G1 导航路径文件缺失，回退为场景查看器展示。"})
            launch_post_generation_display(state, layout_id, "")
            return
        cmd = f"{display_env} bash {G1_WALK_SCRIPT} {layout_id}"
        launch_display_process(state, cmd, "G1 行走可视化")
        return

    # franka / mobile franka → kinematic pick-and-place visualization (GUI loop).
    if robot_type == "mobile franka":
        state.emit("log", {"message": "提示：mobile franka 自动展示使用固定臂抓放可视化（不含底盘移动）。"})
    cmd = (
        f"{display_env} SAGE_LAYOUT_DIR={RESULTS_ROOT / layout_id} SAGE_LAYOUT_ID={layout_id} "
        f"{ISAAC_PYTHON_SH} {FRANKA_VIZ_SCRIPT}"
    )
    launch_display_process(state, cmd, "Franka 抓放可视化")


def run_generation(state: AppState, room_desc: str, robot_type: str = "", room_type: str = "") -> None:
    if robot_type:
        # Robot-task flow: scene + robot + task plan via the robot-aware layout
        # server. The robot client reads SAGE_MAX_TOOL_CALLS from env (no CLI arg).
        cmd = [
            state.generation_python,
            "-u",
            "client_generation_robot_task.py",
            "--room_type",
            room_type or "living room",
            "--robot_type",
            robot_type,
            "--task_description",
            room_desc,
            "--server_paths",
            "../server/layout.py",
        ]
        state.emit("log", {"message": f"开始机器人任务生成（robot={robot_type}, room={room_type or 'living room'}）。"})
        state.emit("log", {"message": f"执行命令：{Path(cmd[0]).name} -u client_generation_robot_task.py --robot_type '{robot_type}' --task_description ... --server_paths ../server/layout.py"})
    else:
        cmd = [
            state.generation_python,
            "-u",  # unbuffered: the client's tool-call counter prints to stdout, which
                   # is block-buffered when piped — without -u the progress line never
                   # arrives until the process exits, so the progress bar never moves.
            "client_generation_room_desc.py",
            "--room_desc",
            room_desc,
            "--server_paths",
            "../server/layout_wo_robot.py",
            "--max_tool_calls",
            str(state.max_tool_calls),
        ]
        state.emit("log", {"message": "开始生成。机器人任务流程已跳过。"})
        state.emit("log", {"message": f"执行命令：{Path(cmd[0]).name} -u client_generation_room_desc.py --room_desc ... --server_paths ../server/layout_wo_robot.py"})
    # The headless kit may have just been (re)started by /api/generate after a GUI
    # display session — block until its MCP port accepts connections.
    if not wait_for_isaac_ready(state):
        state.emit("log", {"message": "Isaac Sim（无头）在超时时间内未就绪，生成中止。"})
        state.emit("status", {})
        return
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "SAGE_MAX_TOOL_CALLS": str(state.max_tool_calls)}
    process = subprocess.Popen(
        cmd,
        cwd=str(CLIENT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )
    state.generation_process = process
    layout_id = None
    if robot_type:
        state.emit("log", {"message": (
            "系统发给 LLM/VLM 的任务内容：\n"
            f"Robot task generation — robot: {robot_type}; room: {room_type or 'living room'}.\n"
            f"Task: {room_desc}"
        )})
    else:
        state.emit("log", {"message": "系统发给 LLM/VLM 的任务内容：\n" + room_generation_instruction(room_desc)})
    state.emit("progress", {"current": 0, "total": state.max_tool_calls})
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        state.emit("log", {"message": line})
        # Drive the progress bar from the client's per-tool-call counter line:
        #   room flow:  "📊 Tool call count: N/M"
        #   robot flow: "🔧 Tool usage: N/M tool calls"
        progress_match = re.search(r"Tool (?:call count|usage):\s*(\d+)\s*/\s*(\d+)", line)
        if progress_match:
            state.emit("progress", {
                "current": int(progress_match.group(1)),
                "total": int(progress_match.group(2)),
            })
        emit_images_from_line(line, state.emit, CLIENT_ROOT)
        emit_images_from_chat_log_line(line, state)
        found = find_layout_id(line)
        if found:
            layout_id = found
            state.latest_layout_id = found
            emit_scene_render_images(found, state)
    code = process.wait()
    state.emit("log", {"message": f"生成进程结束，退出码 {code}。"})
    if code == 0 and layout_id:
        emit_scene_render_images(layout_id, state)
        state.emit("progress", {"current": state.max_tool_calls, "total": state.max_tool_calls, "done": True})
        state.emit("result", {"layout_id": layout_id, "summary": layout_summary(layout_id)})
        try:
            launch_post_generation_display(state, layout_id, robot_type)
        except Exception as exc:
            state.emit("log", {"message": f"GUI 展示启动失败（产物已在结果目录）：{exc}"})
    elif code == 0:
        state.emit("log", {"message": "生成完成，但没有从日志中解析到 layout_id。"})
    else:
        state.emit("log", {"message": "生成失败，请查看上方日志。"})
    state.emit("status", {})


class Handler(SimpleHTTPRequestHandler):
    state: AppState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in {"/api/status", "/status.json"}:
            isaac_status = isaac_service_status(self.state.isaac)
            trellis_status = process_service_status(self.state.trellis)
            self.send_json({
                "isaac": isaac_status,
                "trellis": trellis_status,
                "generation": {"running": self.state.generation_running()},
                "latest_layout_id": self.state.latest_layout_id,
            })
            return
        if path in {"/api/events", "/events"}:
            self.handle_events()
            return
        if path in {"/api/image", "/image-file"}:
            self.serve_image(parsed)
            return
        if path.startswith("/api/layout/") or path.startswith("/layout-json/"):
            layout_id = path.rsplit("/", 1)[-1]
            try:
                self.send_json(latest_layout_result(layout_id))
            except Exception as exc:
                self.send_json({"error": str(exc)}, 404)
            return
        if path.startswith("/results/"):
            return self.serve_result_file(path)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path in {"/api/service", "/service"}:
            data = self.read_json()
            name = data.get("name")
            action = data.get("action")
            proc = getattr(self.state, name, None)
            if name not in {"isaac", "trellis"} or proc is None:
                self.send_json({"ok": False, "message": "未知服务。"}, 400)
                return
            try:
                message = proc.start(self.state.emit) if action == "start" else proc.stop()
                self.state.emit("log", {"message": message})
                self.state.emit("status", {})
                self.send_json({"ok": True, "message": message})
            except Exception as exc:
                self.send_json({"ok": False, "message": str(exc)}, 500)
            return
        if self.path in {"/api/generate", "/generate"}:
            data = self.read_json()
            room_desc = (data.get("room_desc") or "").strip()
            robot_type = (data.get("robot_type") or "").strip()
            room_type = (data.get("room_type") or "").strip()
            allowed_robots = {"", "franka", "mobile franka", "unitree g1"}
            if robot_type not in allowed_robots:
                self.send_json({"ok": False, "message": f"未知机器人类型：{robot_type}"}, 400)
                return
            if self.state.generation_running():
                self.send_json({"ok": False, "message": "已有生成进程正在运行。"}, 409)
                return
            if not room_desc:
                self.send_json({"ok": False, "message": "任务/房间描述不能为空。"}, 400)
                return
            # Reclaim the GPU from any post-generation GUI display, then make sure
            # the headless kit is up (auto-start; run_generation waits for the port).
            self.state.stop_display()
            isaac_status = isaac_service_status(self.state.isaac)
            if not isaac_status["running"]:
                try:
                    msg = self.state.isaac.start(self.state.emit)
                    self.state.emit("log", {"message": f"Isaac Sim 未运行，自动启动（无头）：{msg}"})
                except Exception as exc:
                    self.send_json({"ok": False, "message": f"Isaac Sim 自动启动失败：{exc}"}, 500)
                    return
            thread = threading.Thread(
                target=run_generation,
                args=(self.state, room_desc),
                kwargs={"robot_type": robot_type, "room_type": room_type},
                daemon=True,
            )
            thread.start()
            mode = f"机器人任务（{robot_type}）" if robot_type else "场景生成"
            self.send_json({"ok": True, "message": f"{mode}已开始。"})
            return
        if self.path in {"/api/generate/stop", "/generate/stop"}:
            message = self.state.stop_generation()
            self.state.emit("log", {"message": message})
            self.state.emit("status", {})
            self.send_json({"ok": True, "message": message})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        # Track position by stable event id, NOT list index: the event buffer is
        # trimmed to the last N entries (events[-800:]), so an index-based cursor
        # would stick once the buffer fills and silently stop delivering updates.
        # Resume from Last-Event-ID on reconnect to avoid replaying duplicates.
        try:
            cursor = int(self.headers.get("Last-Event-ID", "0"))
        except (TypeError, ValueError):
            cursor = 0
        # Count this browser connection; when the last one drops, the console
        # auto-shuts everything down after a grace period (see client_disconnected).
        self.state.client_connected()
        try:
            while True:
                with self.state.cond:
                    self.state.cond.wait(timeout=15)
                    events = [e for e in self.state.events if e[0] > cursor]
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                for event_id, event_type, payload in events:
                    self.wfile.write(f"id: {event_id}\n".encode())
                    self.wfile.write(f"event: {event_type}\n".encode())
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    cursor = event_id
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            self.state.client_disconnected()

    def serve_result_file(self, path: str) -> None:
        target = (SERVER_ROOT / path.lstrip("/")).resolve()
        if not path_is_under(target, RESULTS_ROOT) or not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.path = "/" + target.relative_to(SERVER_ROOT).as_posix()
        self.directory = str(SERVER_ROOT)
        return super().do_GET()

    def serve_image(self, parsed) -> None:
        query = parse_qs(parsed.query)
        requested = query.get("path", [""])[0]
        image_path = resolve_local_path(requested)
        if image_path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        event = image_event_for_path(image_path)
        if event is None:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self.path = "/" + image_path.relative_to(SAGE_ROOT).as_posix()
        self.directory = str(SAGE_ROOT)
        return super().do_GET()


def apply_speed_defaults() -> None:
    """Bake in speed defaults that child processes (Isaac, generation) inherit.
    Each uses setdefault, so an explicit env override always wins.

    The dominant cost is the per-placement critic step (Isaac render + VLM call).
    The biggest lever is running it less often:
      - SAGE_CRITIC_FREQUENCY=2: run physics+semantic critics every 2nd placement
        call instead of every call (~halves the heaviest cost).
    Plus cheaper render knobs (near quality-neutral):
      - SAGE_RT_SUBFRAMES=4   : RTX accumulation frames for previews (was 16).
      - SAGE_PREVIEW_RESOLUTION=384 / SAGE_PREVIEW_VIEWS=3 : trim preview renders.

    For an even faster "draft" run (real quality trade-offs) export before launch:
      SAGE_CRITIC_FREQUENCY=3
      SEMANTIC_CRITIC_ENABLED=false       # skip semantic VLM pass entirely (biggest single win)
      PHYSICS_CRITIC_ENABLED=false        # skip physics critic
      SAGE_MAX_TOOL_CALLS=24              # fewer placement rounds (less complete scenes)
      SAGE_DISABLE_TRELLIS=1             # retrieval only, never slow local 3D generation
      SAGE_OBJATHOR_RETRIEVAL_THRESHOLD=24  # accept easier retrieval matches
    """
    # The semantic critic's VLM call is the dominant cost (measured ~30-82s each,
    # growing), so the biggest lever is running it less often. Render is ~1.5s and
    # irrelevant, so rt_subframes/preview matter little — kept modest anyway.
    os.environ.setdefault("SAGE_CRITIC_FREQUENCY", "3")
    # Send only the annotated top-down image to the semantic-critic VLM (0 extra
    # perspective views). That image is what the critic reasons about; dropping the
    # 4 perspective renders cuts vision tokens and a render pass per critic call.
    os.environ.setdefault("SAGE_SEMANTIC_VLM_VIEWS", "0")
    os.environ.setdefault("SAGE_RT_SUBFRAMES", "4")
    os.environ.setdefault("SAGE_PREVIEW_RESOLUTION", "384")
    os.environ.setdefault("SAGE_PREVIEW_VIEWS", "3")


def main() -> None:
    apply_speed_defaults()
    parser = argparse.ArgumentParser(description="SAGE web generation console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--isaac-cmd", default=os.environ.get("SAGE_WEB_ISAAC_CMD", DEFAULT_ISAAC_CMD))
    parser.add_argument("--trellis-cmd", default=os.environ.get("SAGE_WEB_TRELLIS_CMD", DEFAULT_TRELLIS_CMD))
    parser.add_argument("--generation-python", default=DEFAULT_GENERATION_PYTHON)
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    args = parser.parse_args()

    state = AppState(args.isaac_cmd, args.trellis_cmd, args.generation_python, args.max_tool_calls)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    # Let the idle-shutdown timer stop the HTTP server from a background thread.
    state.shutdown_server = server.shutdown

    # Closing the console (Ctrl+C / kill) tears down generation + Isaac + Trellis.
    atexit.register(state.shutdown_all)

    def _signal_shutdown(signum, _frame):
        state.shutdown_all()
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _signal_shutdown)

    print(f"SAGE web console: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown_all()


if __name__ == "__main__":
    main()
