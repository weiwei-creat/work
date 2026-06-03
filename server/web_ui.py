#!/usr/bin/env python3
"""Lightweight web console for SAGE room generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
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

DEFAULT_ISAAC_CMD = "./client/isaac_sim_conda.sh --no-window --experience isaacsim.exp.base.kit --ext-folder /home/gaok/coding/sage/server/isaacsim --enable isaac.sim.mcp_extension"
DEFAULT_TRELLIS_CMD = "bash scripts/start_trellis_server.sh 8080 /home/gaok/coding/TRELLIS trellis5080"
ISAAC_HOST = os.environ.get("SAGE_ISAAC_HOST", "localhost")
DEFAULT_GENERATION_PYTHON = os.environ.get(
    "SAGE_GENERATION_PYTHON",
    str(Path("/home/gaok/anaconda3/envs/sage/bin/python"))
    if Path("/home/gaok/anaconda3/envs/sage/bin/python").exists()
    else sys.executable,
)
DEFAULT_MAX_TOOL_CALLS = int(os.environ.get("SAGE_MAX_TOOL_CALLS", "60"))


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
      display: flex;
      justify-content: center;
      height: calc(100vh - 28px);
      margin: 14px;
      background: transparent;
    }

    .left {
      background: var(--panel);
      border: 1px solid rgba(216,222,231,0.92);
      border-radius: 8px;
      box-shadow: var(--shadow);
      width: min(960px, 100%);
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

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

    @media (max-width: 980px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; }
      .left { min-height: 720px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="left">
      <header>
        <div class="brand-lockup">
          <div class="brand-mark">S</div>
          <div>
            <h1>SAGE 场景对话</h1>
            <div class="sub">用户指令在右侧，LLM/VLM 与工具反馈在左侧</div>
          </div>
        </div>
        <div class="status-row">
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
        <textarea id="prompt" placeholder="描述你想生成的 3D 场景，例如：暖色现代餐厅，橡木桌椅，绿植，开放式置物架，桌面有餐具与细节装饰。"></textarea>
        <div class="controls">
          <div id="runHint" class="hint">Isaac Sim MCP 服务器运行后才能开始生成。</div>
          <div>
            <button id="clearBtn" class="secondary">清空对话</button>
            <button id="generateBtn" disabled>发送生成</button>
          </div>
        </div>
      </div>
    </section>

  </main>

  <script>
    const chat = document.getElementById('chat');
    const promptEl = document.getElementById('prompt');
    const generateBtn = document.getElementById('generateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const isaacToggle = document.getElementById('isaacToggle');
    const trellisToggle = document.getElementById('trellisToggle');
    const isaacLine = document.getElementById('isaacLine');
    const trellisLine = document.getElementById('trellisLine');
    const isaacStateText = document.getElementById('isaacStateText');
    const trellisStateText = document.getElementById('trellisStateText');
    const runHint = document.getElementById('runHint');
    const pendingServices = { isaac: null, trellis: null };
    const pendingSince = { isaac: 0, trellis: 0 };
    let keepChatAtBottom = true;
    let serviceStatus = { isaac: false, trellis: false, generating: false };
    const stages = ['input', 'layout', 'objects', 'export'];
    const shownImageUrls = new Set();

    function setStage(activeStage, completed = []) {
      document.querySelectorAll('.stage').forEach(row => {
        const name = row.dataset.stage;
        const state = row.querySelector('span:last-child');
        row.classList.toggle('active', name === activeStage);
        row.classList.toggle('done', completed.includes(name));
        if (completed.includes(name)) state.textContent = '完成';
        else if (name === activeStage) state.textContent = '进行中';
        else state.textContent = '等待';
      });
    }

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

    function isNearBottom(el, threshold = 50) {
      return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
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

      if (/系统发给\s*LLM\/VLM|发给\s*LLM|发给\s*VLM/i.test(text)) {
        return ['local', '发给 LLM/VLM', text];
      }

      if (/🔤 Response:/i.test(text)) {
        return ['model', 'LLM/VLM 反馈', text.replace(/^.*?🔤 Response:\s*/i, '')];
      }

      if (/Qwen|Claude|OpenAI|LLM|VLM|assistant|thinking|model response|模型反馈|视觉模型|语言模型/i.test(text)) {
        return ['model', 'LLM/VLM 反馈', text];
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

    function isNoiseLog(text) {
      return [
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
      const res = await fetch('/api/status');
      serviceStatus = await res.json();
      updateServiceControl('isaac', serviceStatus.isaac);
      updateServiceControl('trellis', serviceStatus.trellis);
      generateBtn.disabled = !serviceStatus.isaac.running || serviceStatus.generation.running;
      runHint.textContent = serviceStatus.isaac.running
        ? (serviceStatus.generation.running ? '生成正在进行中。' : '可以开始生成。当前流程不会生成机器人任务。')
        : (serviceStatus.isaac.status === 'starting'
          ? 'Isaac Sim 进程已启动，但 MCP 还没有确认可用。'
          : 'Isaac Sim MCP 服务器运行后才能开始生成。');
    }

    async function toggleService(name, checked) {
      const action = checked ? 'start' : 'stop';
      pendingServices[name] = action;
      pendingSince[name] = Date.now();
      updateServiceControl(name, checked);
      addMessage('local', '本地发送', `${name} ${action}`);
      const res = await fetch('/api/service', {
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
    clearBtn.addEventListener('click', () => { chat.innerHTML = ''; });

    generateBtn.addEventListener('click', async () => {
      const roomDesc = promptEl.value.trim();
      if (!roomDesc) {
        addMessage('error', '输入为空', '请输入想要生成的房间描述。');
        return;
      }
      promptEl.value = '';
      addMessage('local', '本地发送', roomDesc);
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room_desc: roomDesc })
      });
      const data = await res.json();
      addMessage(data.ok ? 'server' : 'error', '服务端返回', data.message);
      await refreshStatus();
    });

    const events = new EventSource('/api/events');
    events.addEventListener('log', event => {
      const item = JSON.parse(event.data);
      updateStagesFromLog(item.message || '');
      const chatEvent = chatEventFromLog(item.message || '');
      if (!chatEvent) return;
      const [type, title, body] = chatEvent;
      addMessage(type, title, body, item.time);
    });
    events.addEventListener('result', event => {
      const item = JSON.parse(event.data);
      addMessage('server', '生成完成', `Layout ID: ${item.layout_id}`);
      setStage(null, stages);
    });
    events.addEventListener('image', event => {
      const item = JSON.parse(event.data);
      addImageMessage(item.title || '图片', item.url, item.caption || item.name || '', item.time);
    });
    events.addEventListener('status', () => refreshStatus());

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
    return {
        "url": f"/api/image?path={quote(str(resolved))}",
        "name": rel_name,
        "caption": rel_name,
        "title": title,
    }


def is_scene_render_image(path: Path) -> bool:
    name = path.name.lower()
    return any(
        token in name
        for token in ("rendered_view", "top_down_annotated", "room_", "wall_placement")
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
        if "rendered_view" in name:
            rank = 0
        elif "top_down_annotated" in name:
            rank = 1
        elif name.startswith("room_"):
            rank = 2
        else:
            rank = 3
        return (rank, -path.stat().st_mtime)

    return sorted(unique_images, key=image_priority)


def emit_scene_render_images(layout_id: str, state: AppState, limit: int = 6) -> None:
    for path in scene_render_images_for_layout(layout_id)[:limit]:
        key = str(path)
        if key in state.seen_image_keys:
            continue
        event = image_event_for_path(path)
        if not event:
            continue
        state.seen_image_keys.add(key)
        state.emit("image", event)


def room_generation_instruction(room_desc: str) -> str:
    return (
        "Task: Generate a single room based on the following room description.\n\n"
        f"Room description: {room_desc}\n\n"
        "Process:\n"
        "1. Generate one room layout only.\n"
        "2. Place large furniture first.\n"
        "3. Add functional object groups and detailed surface objects.\n"
        "4. Add decorative/background objects.\n"
        "5. Do not run robot task generation."
    )


def run_generation(state: AppState, room_desc: str) -> None:
    cmd = [
        state.generation_python,
        "client_generation_room_desc.py",
        "--room_desc",
        room_desc,
        "--server_paths",
        "../server/layout_wo_robot.py",
        "--max_tool_calls",
        str(state.max_tool_calls),
    ]
    state.emit("log", {"message": "开始生成。机器人任务流程已跳过。"})
    state.emit("log", {"message": f"执行命令：{Path(cmd[0]).name} client_generation_room_desc.py --room_desc ... --server_paths ../server/layout_wo_robot.py"})
    process = subprocess.Popen(
        cmd,
        cwd=str(CLIENT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    state.generation_process = process
    layout_id = None
    state.emit("log", {"message": "系统发给 LLM/VLM 的任务内容：\n" + room_generation_instruction(room_desc)})
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        state.emit("log", {"message": line})
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
        state.emit("result", {"layout_id": layout_id})
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
        if path == "/api/status":
            isaac_status = isaac_service_status(self.state.isaac)
            trellis_status = process_service_status(self.state.trellis)
            self.send_json({
                "isaac": isaac_status,
                "trellis": trellis_status,
                "generation": {"running": self.state.generation_running()},
                "latest_layout_id": self.state.latest_layout_id,
            })
            return
        if path == "/api/events":
            self.handle_events()
            return
        if path == "/api/image":
            self.serve_image(parsed)
            return
        if path.startswith("/api/layout/"):
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
        if self.path == "/api/service":
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
        if self.path == "/api/generate":
            data = self.read_json()
            room_desc = (data.get("room_desc") or "").strip()
            isaac_status = isaac_service_status(self.state.isaac)
            if not isaac_status["running"]:
                self.send_json({"ok": False, "message": "Isaac Sim MCP 服务器未确认可用，不能开始生成。"}, 409)
                return
            if self.state.generation_running():
                self.send_json({"ok": False, "message": "已有生成进程正在运行。"}, 409)
                return
            if not room_desc:
                self.send_json({"ok": False, "message": "房间描述不能为空。"}, 400)
                return
            thread = threading.Thread(target=run_generation, args=(self.state, room_desc), daemon=True)
            thread.start()
            self.send_json({"ok": True, "message": "生成已开始。"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cursor = 0
        try:
            while True:
                with self.state.cond:
                    self.state.cond.wait(timeout=15)
                    events = self.state.events[cursor:]
                    cursor = len(self.state.events)
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                for _, event_type, payload in events:
                    self.wfile.write(f"event: {event_type}\n".encode())
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

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


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE web generation console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--isaac-cmd", default=os.environ.get("SAGE_WEB_ISAAC_CMD", DEFAULT_ISAAC_CMD))
    parser.add_argument("--trellis-cmd", default=os.environ.get("SAGE_WEB_TRELLIS_CMD", DEFAULT_TRELLIS_CMD))
    parser.add_argument("--generation-python", default=DEFAULT_GENERATION_PYTHON)
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    args = parser.parse_args()

    Handler.state = AppState(args.isaac_cmd, args.trellis_cmd, args.generation_python, args.max_tool_calls)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SAGE web console: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
