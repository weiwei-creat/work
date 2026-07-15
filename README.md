# SAGE: 面向具身AI的可扩展智能体3D场景生成

**Hongchi Xia**, **Xuan Li**, **Zhaoshuo Li**, **Qianli Ma**, **Jiashu Xu**, **Ming-Yu Liu**, **Yin Cui**, **Tsung-Yi Lin**, **Wei-Chiu Ma**, **Shenlong Wang**, **Shuran Song**, **Fangyin Wei**

*NVIDIA, 伊利诺伊大学厄巴纳-香槟分校, 康奈尔大学, 斯坦福大学*

[![论文](https://img.shields.io/badge/论文-PDF-red)](https://arxiv.org/pdf/2602.10116)
[![网站](https://img.shields.io/badge/网站-HTML-green)](https://research.nvidia.com/labs/dir/sage/)
[![数据集](https://img.shields.io/badge/数据集-SAGE--10k-blue)](https://huggingface.co/datasets/nvidia/SAGE-10k)

![Teaser](assets/teaser.png)

## 简介
SAGE 是一个智能体驱动的框架，能够根据用户指定的具身任务，理解任务意图并自动批量生成可直接用于仿真的3D环境。我们发布了场景与动作生成代码，以及由智能体驱动的 SAGE-10k 数据集，以促进后续研究。

## SAGE-10k 数据集

[![数据集](https://img.shields.io/badge/数据集-SAGE--10k-blue)](https://huggingface.co/datasets/nvidia/SAGE-10k)

![预览](assets/preview_and_stats_v2.png)

[SAGE-10k](https://huggingface.co/datasets/nvidia/SAGE-10k) 是一个大规模交互式室内场景数据集，具有逼真的布局，由 SAGE 论文中提出的智能体驱动生成管线生成。该数据集包含 10,000 个多样化场景，涵盖 50 种房间类型和风格，以及 565K 个独立生成的 3D 物体。

## 目录结构

本仓库主要由以下部分组成：

- **`client/`**
  包含客户端实现和脚本。这是用户启动场景生成、控制管线和与 NVIDIA Isaac Sim 交互的主要入口。

- **`server/`**
  承载核心后端逻辑，包括基础模型（LLM、VLM）集成、3D 资产生成（TRELLIS）、材质合成以及场景布局求解器。

- **`IsaacLab/`**
  与 **NVIDIA Isaac Lab** 的集成，为机器人学习和物理交互任务提供仿真环境。
  > **注意：** 此目录包含对原始代码的修改，仍受原始 [BSD-3-Clause License](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE) 约束。

- **`M2T2/`**
  与 **M2T2** 的集成，用于生成接触密集的操控数据并处理复杂的机器人-物体交互。
  > **注意：** 此目录包含对原始代码的修改，仍受原始 [NVIDIA License](https://github.com/NVlabs/M2T2/blob/master/LICENSE) 约束。

- **`matfuse-sd/`**
  与 **MatFuse** 材质生成引擎的集成，用于为 3D 物体和场景生成高质量纹理和材质。
  > **注意：** 此目录包含对原始代码的修改，仍受原始 [MIT License](https://github.com/giuvecchio/matfuse-sd/blob/main/LICENSE) 约束。

- **`robomimic/`**
  与 **robomimic** 的集成，一个用于从示范中学习机器人策略的框架，用于在生成数据上训练策略。
  > **注意：** 此目录包含对原始代码的修改，仍受原始 [MIT License](https://github.com/ARISE-Initiative/robomimic/blob/master/LICENSE) 约束。

## 快速开始

要使用本仓库，您需要同时设置服务端（后端）和客户端（前端/接口）。请参考相应的 README 文件获取详细说明。

### 1. 服务端设置
**[阅读服务端文档](server/README.md)**
  - 后端基础设施的设置说明。
  - VLM（DashScope/Qwen）、LLM（DeepSeek/兼容接口）和 3D 生成模型（TRELLIS）的托管细节。
  - 运行数据增强管线的指南。

### 2. 客户端设置
**[阅读客户端文档](client/README.md)**
  - Python 环境和依赖的安装。
  - 启动和连接 NVIDIA Isaac Sim MCP 服务的说明。
  - 运行场景生成、机器人任务生成和可视化的脚本。

### 使用流程
1.  **启动后端**：按照服务端 README 的说明启动 Isaac Sim MCP，并根据配置选择 Objathor 检索或 TRELLIS 生成。低显存机器默认可不启动 TRELLIS。
2.  **配置客户端**：在客户端目录中设置 `key.json` 和环境变量。
3.  **运行生成**：使用 `client/scripts/` 中的脚本来生成场景（如 `generate_from_room_desc.sh`）或机器人数据。

> Isaac Sim 5.x 的 MCP 服务请使用 `server/README.md` 中的 `./client/isaac_sim_conda.sh --experience isaacsim.exp.base.kit ... --enable isaac.sim.mcp_extension` 启动方式。不要直接启动 `isaacsim.exp.full.kit`，否则容易只打开普通 Isaac Sim 而没有 SAGE MCP socket。

### 关闭 Isaac Sim MCP 服务
如果 Isaac Sim MCP 是在当前终端前台启动的，直接按 `Ctrl+C` 即可退出。

如果服务在后台运行，先查出监听 MCP 端口的进程，再关闭它：
```bash
ss -ltnp | grep 11323
kill <PID>
```

如果使用了自定义端口，例如 `ISAAC_MCP_PORT=11324`，把下面的 `11323` 替换成对应端口。确认关闭成功：
```bash
ss -ltnp | grep 11323
```
没有输出即表示该端口上的 Isaac MCP 服务已停止。

### 低显存资产模式
默认资产来源为 Objathor 现有资产检索，适合 16GB 显存或更小显存的机器：
```bash
export SAGE_OBJATHOR_RETRIEVAL_MODE=embedding
unset SAGE_ENABLE_TRELLIS_GENERATION
```

如需启用 TRELLIS 生成新资产：
```bash
export SAGE_ENABLE_TRELLIS_GENERATION=1
export SAGE_OBJECT_SOURCE=generation
```

可通过以下参数控制资产摆放量：
```bash
export SAGE_OBJECT_QUANTITY_SCALE=0.5
export SAGE_MAX_OBJECTS_PER_TYPE=2
export SAGE_MAX_NEW_OBJECTS_PER_ROOM=20
```

## Web 生成控制台

`server/web_ui.py` 提供一个网页版生成控制台，**左栏对话式展示生成过程**（LLM/VLM 推理与评估反馈实时流 + 4 步进度），**右栏展示生成结果**（场景渲染画廊、灯箱预览、布局概要卡）。还能在页面上一键开关 Isaac Sim / Trellis 服务。

启动（用 `sage` 环境，**不要 sage5080**）：

```bash
conda activate sage
cd server
# 绑 0.0.0.0 即可从本机浏览器经局域网/Tailscale 打开
python web_ui.py --host 0.0.0.0 --port 7860
# 然后浏览器访问 http://<服务器IP>:7860  （例如 http://100.111.153.8:7860）
```

若只在带桌面的服务器本机查看，可绑回环地址并用远程桌面（如 SunLogin）打开：

```bash
python web_ui.py --host 127.0.0.1 --port 7860   # 浏览器打开 http://127.0.0.1:7860
```

界面功能（近期新增）：
- **左栏对话**只显示精简要点（过滤路径/参数/目录等噪声）；**底部「命令行 log」**面板原样展示后台全部输出，方便看进度。两栏都默认跟随最新。
- **生成计时器**（右上 ⏱）：发送即计时，结束停止。
- **工具调用进度条 + 预计剩余**：按客户端 `Tool call count: N/M` 估算进度与 ETA。
- **大图预览自动跟随最新生成图**；点缩略图可回看。
- **「终止」按钮**：随时结束并清空当前任务，**不影响** Isaac/Trellis。
- **关闭控制台即全部关停**：`Ctrl+C`/`pkill web_ui.py` 或关闭浏览器页面（宽限期无重连）都会连带停掉 Isaac、Trellis、生成进程（`SAGE_WEB_IDLE_SHUTDOWN_SECONDS` 调宽限期，设 0 禁用）。

常用参数 / 说明：
- `--port`（默认 7860）、`--host`（默认 127.0.0.1）
- `--max-tool-calls`（默认 **40**，控制单次生成丰富度；多房间需更多轮数，模型常在用满前自行停止）
- 控制台默认走**无机器人**流程（`client_generation_room_desc.py` + `server/layout_wo_robot.py`），**按描述自动出单房间或多房间/整屋**：如"两居室公寓,带客厅和厨房"会生成多个连通房间并逐个装饰;"一个厨房"则只出一间。（底层仍是单房间 server,因其原生支持多房间且带本仓库的全部提速/保护修复;未用 `layout_wo_robot_multiroom.py`。）
- **机器人任务模式**：输入框上方「机器人」下拉框选 Franka（固定，抓放）/ 移动 Franka（导航+抓放）/ 宇树 G1（仅导航），并填房间类型；此时文本框写**任务描述**（建议英文，如 "the robot must pick up the cube from the coffee table and place it on the desk"），后台改走 `client_generation_robot_task.py` + `server/layout.py`（含任务解析与可行性矫正）。进度条/终止按钮同样适用。
- 启动时 `apply_speed_defaults()` 注入低风险提速默认（`SAGE_CRITIC_FREQUENCY=3`、`SAGE_SEMANTIC_VLM_VIEWS=0`、`SAGE_RT_SUBFRAMES=4`、`SAGE_PREVIEW_VIEWS=3/RESOLUTION=384`），显式 export 可覆盖。
- 需要 Isaac Sim MCP 服务在运行才能开始生成——可直接用页面左上角的「Isaac Sim」开关启动。
- **运维细节与远程部署（含 B200 自托管 Qwen3-VL）见 [`CLAUDE.md`](CLAUDE.md)。**

## 无机器人场景生成（单房间 / 多房间）

除了机器人任务场景，仓库还能生成**纯场景（不放机器人、不做任务解析/可行性矫正）**，分单房间和多房间两种入口：

| 用途 | 客户端 | server |
|------|--------|--------|
| 单房间 · 无机器人 | `client/client_generation_room_desc.py` | `server/layout_wo_robot.py` |
| 多房间 · 无机器人 | `client/client_generation.py` | `server/layout_wo_robot_multiroom.py` |

> **[重要] 用 `sage` 环境，不要用 `sage5080`。** layout server 是纯 CPU 程序，通过 socket 连 Isaac kit，本身不需要 Isaac。而 `sage5080` 的 conda 激活脚本会把 Isaac 的 `pip_prebundle` 注入 `PYTHONPATH`，其中为 Isaac Python 3.11 编译的 `matplotlib` 会覆盖环境自己的，导致 `room_solver.py` 在 `import matplotlib.pyplot` 时报循环导入错误（`ImportError: cannot import name '_c_internal_utils'`）。`sage` 环境的 matplotlib 是干净的。

> 注意：如果已有一个 Isaac MCP kit 在跑（监听 11323），**不要再启动第二个**（脚本里那段 `while true; do ... isaac_sim_conda.sh ...` 是用来起 kit 的）。直接跑下面的 python 客户端即可，它会连现有 kit。

### 单房间 · 无机器人

```bash
conda activate sage
cd client
python client_generation_room_desc.py \
    --room_desc "A medium-sized kitchen." \
    --server_paths ../server/layout_wo_robot.py
    # 物体多时可加 --max_tool_calls 30
```

`--room_desc` 可自由描述风格/大小，例如 `"A van gogh starry-night style bedroom."`、`"A rusty abandoned restroom."`。

### 多房间 · 无机器人

```bash
conda activate sage
cd client
python client_generation.py \
    --input_prompt_path prompts/multi_room/craft.txt \
    --server_paths ../server/layout_wo_robot_multiroom.py
```

多房间用整篇 prompt 文件，仓库自带 4 个：`client/prompts/multi_room/` 下的 `1b1b_phd_apartment.txt`、`craft.txt`、`mid_century_family.txt`、`naturalist.txt`。自定义时复制一个 `.txt`，改第一行 `Task: Generate a layout of "<你的描述>" including multiple rooms.` 即可。

两种生成的产物都在 `server/results/<layout_id>/`。

## 宇树 G1 导航任务（生成 + 行走可视化）

除了移动式 Franka 操作任务，本仓库还支持生成**宇树 G1 人形机器人的导航任务**：在生成的房间里规划一条无碰撞路径，并让 G1 沿路径"行走"进行运动学可视化（出图/视频）。

整个流程分两个阶段，**两个阶段的 Python 环境不同**：

| 阶段 | 脚本 | Python 环境 | 是否需要 conda |
|------|------|-------------|----------------|
| A. 场景生成 + 路径规划 | `client/client_generation_robot_task.py` | conda `sage` | **需要** |
| B. G1 行走渲染 | `server/isaacsim/isaac.sim.mcp_extension/examples/g1_walk_render.py` | Isaac Sim 自带 `python.sh` | **不需要** |

### 阶段 A：生成 G1 导航场景（需要 conda sage）

确保 Isaac Sim MCP 服务已启动（见上文），然后用 `unitree g1` 作为机器人类型运行客户端：

```bash
conda activate sage          # 阶段 A 需要 conda
cd client
python client_generation_robot_task.py \
    --room_type "living room" \
    --robot_type "unitree g1" \
    --task_description "A Unitree G1 humanoid must navigate from the center of the room to the sofa, then walk to a bookshelf, keeping a clear collision-free path between the landmarks." \
    --server_paths ../server/layout.py
```

解析工具会自动识别 `robot_type=unitree_g1`（纯导航，只产生 `navigate` 步骤），可行性矫正会输出：

- 场景资产：`server/results/<layout_id>/<layout_id>_usd_collection/`
- 行走路径：`server/results/<layout_id>/<layout_id>_g1_nav_path.json`（无碰撞路点，既是可行性检查，也是阶段 B 要跟随的轨迹）

### 阶段 B：渲染 G1 沿路径行走（不需要 conda）

阶段 B 用 Isaac Sim 自带的 Python（`python.sh`）独立启动，**不要** activate conda（它只依赖 `numpy` + `omni.isaac.core`，且独立于无头 MCP kit，不会互相干扰）。

**最简单：一键脚本，只需 `layout_id`。** 默认**窗口模式 + 循环行走**：

```bash
cd <SAGE_ROOT>
./server/isaacsim/isaac.sim.mcp_extension/examples/run_g1_walk.sh <layout_id>
```

- 不带参数会自动选用 `server/results/` 里**最新**的场景
- 默认窗口模式（`DISPLAY=:0`），在生成的房间里**循环行走**，关闭窗口即退出（用 SunLogin 等远程桌面观看）
- 想离屏渲染成视频，加 `--headless`：

```bash
./server/isaacsim/isaac.sim.mcp_extension/examples/run_g1_walk.sh <layout_id> --headless
# 产物：/tmp/g1_render_<layout_id>/frames.mp4 + 逐帧 PNG
```

脚本会自动解析所有路径、定位 Isaac Sim 并用 `python.sh` 启动，无需手动设置环境变量、无需 conda。

**可选微调**（运行脚本前 `export` 即可，脚本会沿用）：

```bash
export ISAAC_SIM_PATH=$HOME/coding/isaacsim   # Isaac Sim 安装根目录（默认即此）
export SAGE_WALK_LOOPS=3                       # 行走遍数（窗口模式默认无限循环）
export SAGE_G1_USD=<path-or-url>               # 默认指向 Isaac 5.1 自带的带腿 G1（37 dof，含 hip/knee）
export SAGE_CAM_BACK=1.2                       # 相机离近墙的距离（米，越大看得越全）
export SAGE_CAM_UP=3.0                         # 相机高于天花板的高度（米，越大越俯视）
export SAGE_CAM_TARGET_Z=0.4                   # 相机注视点高度（米）
export SAGE_RENDER_OUT=/tmp/my_out             # 渲染输出目录（默认 /tmp/g1_render_<layout_id>）
```

> 提示：默认会自动隐藏天花板，方便从上方看到室内。`/tmp` 下的输出重启会清空，需要保留就拷到 `server/results/<layout_id>/` 下。

<details>
<summary>手动方式（不用一键脚本时）</summary>

```bash
R=<ISAAC_SIM_PATH>/_build/linux-x86_64/release
export SAGE_LAYOUT_DIR=<SAGE_ROOT>/server/results/<layout_id>
export SAGE_LAYOUT_ID=<layout_id>
export SAGE_RENDER_HEADLESS=0        # 0=窗口, 1=无头出mp4
export DISPLAY=:0                    # 窗口模式需要
export XAUTHORITY=$HOME/.Xauthority
$R/python.sh server/isaacsim/isaac.sim.mcp_extension/examples/g1_walk_render.py
```
</details>

## 场景生成质量改进

参考 SceneSmith (arXiv:2602.09153) 对生成 prompt / 流程做了一组改进（更高物体密度、
放置回滚、量化评测等）。详见 **[docs/scene_generation_improvements.md](docs/scene_generation_improvements.md)**。

常用开关：

```bash
export SAGE_MAX_TOOL_CALLS=15          # 单次生成最大工具调用数（越大场景越丰富）
export SAGE_ENABLE_ROLLBACK=true       # 放置使房间变差时自动回滚
export SAGE_MIN_ITEMS_PER_SURFACE=2    # 每个支撑面至少摆几件小物
```

提速 / 质量开关（瓶颈实测为「每次摆放的语义评审 VLM 调用」，渲染仅约 1.5s）：

```bash
export SAGE_CRITIC_FREQUENCY=3         # 物理+语义评审每 N 次摆放才跑一次（最大提速杠杆）
export SAGE_SEMANTIC_VLM_VIEWS=0       # 语义评审只发标注俯视图 1 张（省视觉 token）
export SAGE_ALLOW_REMOVE_REQUESTED=true  # 允许评审移除用户点名物体（默认 false：保护不删）
export SAGE_MAX_TOKENS=16384           # 客户端单次输出上限（默认 16384，省上下文）
export SAGE_ISAAC_DISPLAY=:0           # GUI 展示使用的 X display
export SAGE_TASK_PICK=apple            # franka 可视化抓取物（物体 id 子串）
export SAGE_TASK_PLACE=plate           # franka 可视化放置目标
```

> **上下文超限防护**：客户端会压缩较旧的工具结果（保留最近 `SAGE_FULL_TOOL_RESULT_WINDOW=6` 条全文，更旧的截到 `SAGE_TOOL_RESULT_TRUNCATE_CHARS=600` 字符）；若仍触发模型上下文上限错误，会自动减半压缩参数并重试而非中断。自托管 vLLM 的 `--max-model-len` 已设为 131072。
> **生成无头、展示带窗**：生成期间 Isaac 以无头模式运行（更快更省显存）；生成结束后控制台自动停掉无头 kit 并在工作站桌面弹出 **GUI 展示窗口（自动带灯光、隐藏天花板）**——纯场景弹场景查看器；franka/mobile franka 任务弹**抓放任务可视化**（机械臂 IK 抓起目标物放到指定位置，循环重放）；G1 任务弹行走可视化。下一次点「发送生成」会自动关掉展示窗口并按需拉起无头 Isaac（无需手动开关）。

> 用户在描述里明确点名的物体（如 "a basketball and a soccer ball"）**默认受保护**，语义评审不会建议或执行删除（`is_user_requested_object`，layout_wo_robot.py）。
> 自托管 VLM（把 Qwen3-VL 部署到 B200、SAGE 改走本地 vLLM）可把语义评审从 ~58–100s 降到 ~9–10s、整轮 22min→7.6min，详见 [`CLAUDE.md`](CLAUDE.md) 第 8 节。

### 一键拉起 B200 自托管 Qwen3-VL

5080 上有运维脚本 `~/sage_b200/up.py`（**不在本仓库内**，避免主机/凭据逻辑入库），幂等，做三件事：检查/启动 B200 上的 vLLM → 起本地 SSH 隧道（`127.0.0.1:8200` → pod `:8000`，断线自动重连）→ 轮询 `http://127.0.0.1:8200/v1/models` 直到就绪。

```bash
# 在 5080 上运行；密码来源二选一：环境变量 B200_PASSWORD，或 ~/.sage_b200_pass（chmod 600），不要写进仓库
B200_PASSWORD='<B200密码>' /home/gaok/anaconda3/envs/sage/bin/python ~/sage_b200/up.py

curl http://127.0.0.1:8200/v1/models        # 返回 qwen3-vl 即就绪
pkill -f 'sage_b200/up.py --tunnel'         # 只停隧道（B200 上的 vLLM 不受影响）
```

> 冷启动需等权重加载 + FP8-MoE 内核 JIT（可能十几分钟）；SAGE 侧通过 `.env` 的 `QWEN_BASE_URL=http://127.0.0.1:8200/v1` 接入，切回 DashScope 用 `.env.dashscope.bak`。详见 [`CLAUDE.md`](CLAUDE.md) 第 8 节。

离线评测一个已生成场景的质量（无需 Isaac）：

```bash
conda activate sage
python server/eval_scene_metrics.py --layout_id <layout_id>
# 输出 CNT/COL/OOB/NAV/ACC/STB 指标
```

## 引用

如果您觉得我们的工作对您的研究有帮助，请考虑引用：

```bibtex
@article{xia2026sage,
  title={SAGE: Scalable Agentic 3D Scene Generation for Embodied AI},
  author={Xia, Hongchi and Li, Xuan and Li, Zhaoshuo and Ma, Qianli and Xu, Jiashu and Liu, Ming-Yu and Cui, Yin and Lin, Tsung-Yi and Ma, Wei-Chiu and Wang, Shenlong and Song, Shuran and Wei, Fangyin},
  journal={arXiv preprint arXiv:2602.10116},
  year={2026}
}
```


## 致谢

我们衷心感谢以下项目的作者，感谢他们奠基性的工作和开源贡献。本仓库构建并适配了以下项目的组件：

对以下组件做了修改，它们仍受其原始许可证约束：

| 仓库 | 许可证 |
|------------|---------|
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | [BSD-3-Clause](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE) |
| [NVlabs/M2T2](https://github.com/NVlabs/M2T2) | [NVIDIA License](https://github.com/NVlabs/M2T2/blob/master/LICENSE) |
| [giuvecchio/matfuse-sd](https://github.com/giuvecchio/matfuse-sd) | [MIT License](https://github.com/giuvecchio/matfuse-sd/blob/main/LICENSE) |
| [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | [MIT License](https://github.com/ARISE-Initiative/robomimic/blob/master/LICENSE) |

此外，我们的 MCP 客户端、服务端和机器人仿真的实现借鉴和参考了：

| 仓库 | 许可证 |
|------------|---------|
| [allenai/Holodeck](https://github.com/allenai/Holodeck) | [Apache 2.0](https://github.com/allenai/Holodeck/blob/main/LICENSE) |
| [xiahongchi/HoloScene](https://github.com/xiahongchi/HoloScene) | [Apache 2.0](https://github.com/xiahongchi/HoloScene/blob/master/LICENSE.txt) |
| [Entongsu/DRAWER-Real2Sim2Real](https://github.com/Entongsu/DRAWER-Real2Sim2Real) | [Apache 2.0](https://github.com/black-forest-labs/flux/blob/main/LICENSE) |
| [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | [MIT License](https://github.com/microsoft/TRELLIS/blob/main/LICENSE) |
| [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | [Apache 2.0](https://github.com/black-forest-labs/flux/blob/main/LICENSE) |
| [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | [Apache 2.0](https://github.com/QwenLM/Qwen3?tab=readme-ov-file#license-agreement) |
