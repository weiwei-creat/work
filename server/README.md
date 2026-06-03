# MCP 服务端：场景生成后端

此仓库包含 MCP 场景生成管线的服务端实现。

## 0. 环境选择

SAGE 提供以下 conda 环境，根据你的 GPU 选择：

| 环境 | PyTorch | CUDA | Python | Isaac Sim | 适用 GPU |
|------|---------|------|--------|-----------|----------|
| `env_isaaclab` | 2.8.0+cu128 | 12.8 | 3.11 | 5.1（直接启动） | RTX 5080 (Blackwell) |
| `sage` | 2.5.1+cu124 | 12.4 | 3.10 | —（仅 MCP 客户端） | RTX 30/40 系列 |
| `sage5080` | 2.5.1+cu124 | 12.4 | 3.10 | —（仅 MCP 客户端） | RTX 5080（MCP 客户端） |

**关键区别**：
- `env_isaaclab` 是 IsaacLab 数据生成的**唯一可用环境**（需要 Python 3.11 + PyTorch 2.8+）
- `sage` / `sage5080` 用于场景生成客户端与 Isaac Sim MCP wrapper；IsaacLab 数据生成仍必须使用 `env_isaaclab`

### 配置 env_isaaclab（IsaacLab 数据生成）

```bash
conda activate env_isaaclab

# IsaacLab 扩展安装（首次）
pip install -e /home/gaok/coding/sage/IsaacLab/source/extensions/omni.isaac.lab
pip install -e /home/gaok/coding/sage/IsaacLab/source/extensions/omni.isaac.lab_assets
pip install -e /home/gaok/coding/sage/IsaacLab/source/extensions/omni.isaac.lab_tasks

# M2T2 依赖
pip install -e /home/gaok/coding/sage/M2T2/pointnet2_ops
```

## 1. 准备与设置

### 1.0 背景布局数据准备
SAGE 从 objathor 数据集中获取门的纹理。请按照以下说明操作（复制自 Holodeck: https://github.com/allenai/Holodeck/blob/main/README.md），你可能需要安装 objathor 包：

运行以下命令下载数据：
```bash
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23
```
默认情况下，这些数据会保存到 `~/.objathor-assets/...`。

你可能还需要门的材质和纹理坐标，下载链接：https://drive.google.com/file/d/1frgSjaYj7rp_CDcKOVTuvbCl6St1W6Fo/view?usp=drive_link

### 1.1 启动 TRELLIS 服务
```bash
cd /home/gaok/coding/sage
bash scripts/start_trellis_server.sh 8080 /home/gaok/coding/TRELLIS trellis5080
```
下载模型检查点需要 Hugging Face Token（`HF_TOKEN`）。

检查 TRELLIS 服务是否启动成功：
```bash
curl http://127.0.0.1:8080/health
```

### 1.2 启动 Isaac Sim MCP 服务端
```bash
cd /home/gaok/coding/sage
conda activate sage5080
./client/isaac_sim_conda.sh \
  --no-window \
  --experience isaacsim.exp.base.kit \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```
启动成功后日志中应能看到 MCP 端口，例如：
`Isaac Sim MCP server started on localhost:11323`。
当前默认端口由 `SLURM_JOB_ID` 哈希得到；非 SLURM 环境下通常是 `11323`。以启动日志为准。

确认 MCP 端口正在监听：
```bash
ss -ltnp | grep 11323
```

注意：直接启动 `isaacsim.exp.full.kit` 只能打开普通 Isaac Sim，不会启动 SAGE 的 MCP socket。预览图和 critic 流程必须使用上面的 `--ext-folder ... --enable isaac.sim.mcp_extension` 启动方式。源码版 Isaac Sim 的 `isaac-sim.sh` 默认会强制加载 Full experience，SAGE 的 `client/isaac_sim_conda.sh` 会直接调用 `kit/kit` 来确保 `--experience isaacsim.exp.base.kit` 生效。

Isaac MCP 的默认碰撞近似为 `convexDecomposition`，可通过环境变量覆盖：
```bash
SAGE_COLLISION_APPROXIMATION=convexHull ./client/isaac_sim_conda.sh ...
```
不要在常规生成里使用 `sdf`。如果日志出现 `cudaErrorIllegalAddress`、`createSDFBuilder` 或 PhysX GPU SDF cooking 相关崩溃，优先确认没有覆盖成 `sdf`，并确认启动日志路径是 `Isaac-Sim Base/5.1` 而不是 `Isaac-Sim Full/5.1`。

### 1.2.1 Isaac 预览图验证
预览图只允许使用 Isaac Sim/Replicator 渲染；CPU fallback 已关闭。Isaac MCP 未启动、渲染失败、生成全黑/全白图时会直接报错。

无机器人房间生成在 physics critic 成功后会默认自动生成 Isaac 预览图：
```bash
server/results/<layout_id>/preview/<room_id>_rendered_view_*.png
```
可用环境变量控制：
```bash
export SAGE_RENDER_PREVIEW=1          # 默认 1；设为 0 可跳过自动预览图
export SAGE_PREVIEW_RESOLUTION=512    # 默认 512
export SAGE_PREVIEW_VIEWS=4           # 默认 4
export SAGE_REQUIRE_ISAAC_PREVIEW=1   # 默认 1；预览失败时让生成报错
```

语义 critic 会优先把同一批 Isaac Sim/Replicator 渲染图作为 VLM 的 perspective 输入，并复制到：
```bash
server/vis/<room_id>_rendered_view_*.png
```
对应开关：
```bash
export SAGE_USE_ISAAC_RENDER_FOR_VLM=1     # 默认 1
export SAGE_REQUIRE_ISAAC_VLM_RENDERS=1    # 默认 1；VLM 渲染图缺失时让 critic 报错
```

手动验证某个 layout 的预览图：
```bash
cd /home/gaok/coding/sage
ISAAC_MCP_PORT=11323 python server/isaaclab/layout_preview.py \
  --layout_id layout_7c95fb4c \
  --resolution 256
```

成功后应生成：
```bash
ls server/results/layout_7c95fb4c/preview/*_rendered_view_*.png
```

如果报 `ConnectionRefusedError`，先确认 MCP 端口监听和 Isaac 启动命令；如果报 invalid preview，检查 Isaac 日志中的 `render_room_preview`、`Replicator`、`BasicWriter` 相关错误。

如果 Isaac 启动时报 `Address already in use`，说明同一端口已经有一个 Isaac MCP 服务在运行。先确认并清理旧进程：
```bash
ss -ltnp | grep 11323
```
也可以显式换端口启动，并让客户端使用同一个端口：
```bash
ISAAC_MCP_PORT=11324 ./client/isaac_sim_conda.sh \
  --no-window \
  --experience isaacsim.exp.base.kit \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```
客户端命令也需要带同一个端口，例如 `ISAAC_MCP_PORT=11324 python server/isaaclab/layout_preview.py ...`。


### 1.3 视觉语言模型（VLM）
当前使用阿里云 DashScope API（`qwen-vl-max-latest`），无需本地启动。配置见 `.env` 文件。

如需自托管，可使用 **Qwen3-VL** + vLLM：
```bash
# 下载模型
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir /tmp/Qwen3-VL-30B-A3B-Instruct

# 启动模型
cd /tmp
vllm serve Qwen3-VL-30B-A3B-Instruct \
    --port 8080 \
    --max-model-len 32768 \
    --async-scheduling \
    --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}' \
    --mm-processor-cache-gb 0
```

### 1.4 通用大语言模型（LLM）
当前使用 DeepSeek API（`deepseek-v4-pro`，通过 Anthropic 兼容接口），无需本地启动。配置见 `.env` 文件。

如需自托管，可使用 **gpt-oss-120b** + vLLM：
```bash
vllm serve openai/gpt-oss-120b --port 8080 --tensor-parallel-size 4 --async-scheduling
```

### 1.5 配置
在 `.env` 或 `key.json` 中填入必要的 API 密钥和 URL。配置加载优先级：环境变量 > `.env` > `server/key.json`。

### 1.6 资产生成与检索配置
默认情况下，SAGE 不调用 TRELLIS 生成新 3D 资产，而是从本地 Objathor 资产库检索已有物体。这样更适合 16GB 显存或更小显存的机器。

Objathor 检索默认使用 CLIP/SBERT embedding：
```bash
export SAGE_OBJATHOR_RETRIEVAL_MODE=embedding
```

如果需要更轻量的 CPU 文本检索回退，可改为：
```bash
export SAGE_OBJATHOR_RETRIEVAL_MODE=text
```

如需启用 TRELLIS 生成，必须显式开启，并确保终端 1 的 TRELLIS 服务已经启动：
```bash
export SAGE_ENABLE_TRELLIS_GENERATION=1
```

如需强制使用 Objathor 检索，即使 TRELLIS 开关已开启，也可以设置：
```bash
export SAGE_OBJECT_SOURCE=objaverse
```

兼容旧配置：
```bash
export SAGE_DISABLE_TRELLIS=1
```

资产摆放量可通过以下参数控制：
```bash
export SAGE_OBJECT_QUANTITY_SCALE=0.5      # 按比例缩放每类资产数量
export SAGE_MAX_OBJECTS_PER_TYPE=2         # 每种物体最多放几个，0 表示不限
export SAGE_MAX_NEW_OBJECTS_PER_ROOM=20    # 每轮每个房间最多新增几个，0 表示不限
```

## 2. 运行生成

以下三个终端需按顺序启动。

### 终端 1：TRELLIS 服务端（可选）
仅当设置 `SAGE_ENABLE_TRELLIS_GENERATION=1` 时需要启动 TRELLIS。默认 Objathor 检索模式不需要启动此服务。

```bash
cd /home/gaok/coding/sage
bash scripts/start_trellis_server.sh 8080 /home/gaok/coding/TRELLIS trellis5080
```
检查是否启动成功：
```bash
curl http://127.0.0.1:8080/health
```

### 终端 2：Isaac Sim MCP 服务端
```bash
cd /home/gaok/coding/sage
conda activate sage5080
./client/isaac_sim_conda.sh \
  --no-window \
  --experience isaacsim.exp.base.kit \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```
启动成功后日志中应能看到 MCP 端口，例如：
`Isaac Sim MCP server started on localhost:11323`。
当前默认端口由 `SLURM_JOB_ID` 哈希得到；非 SLURM 环境下通常是 `11323`。以启动日志为准。

确认 MCP 端口正在监听：
```bash
ss -ltnp | grep 11323
```

注意：直接启动 `isaacsim.exp.full.kit` 只能打开普通 Isaac Sim，不会启动 SAGE 的 MCP socket。预览图和 critic 流程必须使用上面的 `--ext-folder ... --enable isaac.sim.mcp_extension` 启动方式。源码版 Isaac Sim 的 `isaac-sim.sh` 默认会强制加载 Full experience，SAGE 的 `client/isaac_sim_conda.sh` 会直接调用 `kit/kit` 来确保 `--experience isaacsim.exp.base.kit` 生效。

### 终端 3：运行 SAGE 后端 / 生成端
低显存默认运行方式：
```bash
export SAGE_OBJATHOR_RETRIEVAL_MODE=embedding
unset SAGE_ENABLE_TRELLIS_GENERATION
```

需要 TRELLIS 生成时：
```bash
export SAGE_ENABLE_TRELLIS_GENERATION=1
export SAGE_OBJECT_SOURCE=generation
```

**无机器人房间生成：**
```bash
cd /home/gaok/coding/sage/client
conda activate sage       # RTX 5080 用户请用 conda activate sage5080
python client_generation_room_desc.py \
  --room_desc "A bedroom." \
  --server_paths ../server/layout_wo_robot.py
```

**机器人任务生成：**
```bash
cd /home/gaok/coding/sage/client
conda activate sage       # RTX 5080 用户请用 conda activate sage5080
python client_generation_robot_task.py \
  --room_type "bedroom" \
  --robot_type "mobile franka" \
  --task_description "In a bedroom, the robot must pick up the water bottle from the nightstand and place it on the desk." \
  --server_paths ../server/layout.py
```

## 3. 数据增强

### 3.1 通用姿态增强
对场景中的小型置物类物体进行姿态增强，使用以下脚本：
```bash
./augment/scripts/general_pose_augmentation.sh
```

### 3.2 通用物体类别级增强
对场景中的物体进行类别级增强，使用以下脚本：
```bash
./augment/scripts/general_cat_augmentation.sh
```

## 4. 机器人数据生成

### 4.1 前置条件
确保已安装 **IsaacLab** 和 **M2T2**。请参考 `../IsaacLab` 和 `../M2T2` 中各自的安装指南。

### 4.2 固定式 Franka 机械臂任务
将物体类别级增强应用于机器人数据生成：
```bash
./augment/scripts/robot_data_generation_franka_arm.sh
```

### 4.3 移动式 Franka 任务
此管线支持对机器人数据生成进行场景布局级增强。目前支持自动生成一轮**导航 + 抓取放置**的数据。

**选项 A：仅姿态增强（任务相关物体）**
```bash
./augment/scripts/robot_data_generation_mobile_franka.sh
```

**选项 B：场景布局级增强**
1.  **生成布局增强：** 请参考 `../client/` README 中的布局生成说明。
2.  **生成带姿态增强的数据：**
```bash
./augment/scripts/robot_data_generation_mobile_franka_scene_aug.sh
```

## 5. 策略训练

有关策略训练的说明，请参考 `../robomimic` 文档，并使用生成的 HDF5 数据。

---

## 6. Isaac Sim 5.1 兼容性（2026-05-20 更新）

SAGE 已适配 Isaac Sim 5.1。以下修改已应用到 `../IsaacLab/` 源码中，解决了旧 IsaacLab 在 5.x 下的若干不兼容问题。

### 6.1 关键修复

| 文件 | 修改 | 原因 |
|------|------|------|
| `omni.isaac.lab/app/app_launcher.py` | `from isaacsim import SimulationApp` | 5.x 中 `omni.isaac.kit` 已废弃，直接导入会导致 `ModuleNotFoundError: omni.kit.usd` |
| `omni.isaac.lab/app/app_launcher.py` | 新增 `/isaaclab/cameras_enabled` carb setting | 不再依赖 `.kit` 文件中的 `[settings.isaaclab]` 来启用相机 |
| `omni.isaac.lab/sim/converters/urdf_converter.py` | 新增 `_fix_physx5_compatibility()` | 5.x URDF importer 产生 instanceable prim + 退化质量/惯性，导致 PhysX 5.x 在 `timeline.commit()` 时卡死 |
| `omni.isaac.lab/sim/converters/urdf_converter.py` | `set_import_option()` 兼容层 | 5.x URDF ImportConfig API 从 setter 方法变为属性 |
| `omni.isaac.lab/sim/converters/urdf_converter.py` | `isaacsim.asset.importer.urdf` 回退 | 5.x 中扩展名从 `omni.importer.urdf` 改名 |
| `omni.isaac.lab/sensors/camera/camera.py` | `__init__` 中预初始化 output dict | ObservationManager 在 simulation play 前查询 camera shape，5.x 下 `_is_outdated` 未就绪 |
| `omni.isaac.lab/sensors/sensor_base.py` | `_update_outdated_buffers` 加 `hasattr` guard | 防止 sensor 未初始化时访问 `_is_outdated` |
| `omni.isaac.lab/envs/ui/base_env_window.py` | `omni.isaac.ui` 缺失回退 | 5.x 移除了 `omni.isaac.ui` 模块，GUI 模式下 UI 窗口降级为空 |
| `omni.isaac.lab/envs/manager_based_env.py` | UI 窗口创建 try-except | GUI 模式下 `omni.isaac.ui` 不可用时跳过控制面板 |
| `source/apps/isaaclab.python.headless.kit` | 依赖更新为 5.x 命名 | `omni.isaac.kit` → `isaacsim.simulation_app` 等 |
| `server/isaaclab/data_generation_*.py` | `num_envs` 作用域修复 | 全局变量与 CLI 参数冲突导致访问不存在的 `/World/envs/env_1` |

### 6.2 RTX 5080 (Blackwell) 环境

RTX 5080 需要 PyTorch ≥ 2.7（支持 sm_120）。`env_isaaclab` 环境已升级：

```bash
conda activate env_isaaclab
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# 预期: 2.8.0+cu128  NVIDIA GeForce RTX 5080
```

安装 PyTorch 2.8.0 的方法（wheel 文件需手动下载）：

```bash
# 1. 下载 wheel 文件（约 2.5 GB），链接见下方
# 2. 全部放到 ~/下载/ 后执行：
pip install ~/下载/*.whl
# 3. 重编译 CUDA 扩展
pip install --no-build-isolation -e /home/gaok/coding/tigon/external/nvdiffrast
pip install --no-build-isolation -e /home/gaok/coding/sage/M2T2/pointnet2_ops
```

所需 wheel 文件列表：
| 包 | 大小 | 链接 |
|----|------|------|
| torch 2.8.0+cu128 | ~800 MB | `https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl` |
| torchvision 0.23.0+cu128 | ~9 MB | `https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl` |
| torchaudio 2.8.0+cu128 | ~4 MB | `https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl` |
| triton 3.4.0 | ~155 MB | `https://download.pytorch.org/whl/triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` |
| NVIDIA 依赖 (×9) | ~1.4 GB | `https://pypi.nvidia.com/nvidia-{cublas/cudnn/cufft/curand/cusolver/cusparse/cusparselt/nccl/nvjitlink}-cu12/*.whl` |

> **注意**：首次运行前需清除 URDF 转换缓存 `rm -rf /tmp/IsaacLab/usd_*`，以确保 PhysX 5.x 兼容修复生效。

### 6.3 M2T2 模型

M2T2 模型权重需从 HuggingFace 下载（约 131 MB）：

```bash
wget -O /home/gaok/coding/sage/M2T2/m2t2.pth \
  "https://huggingface.co/wentao-yuan/m2t2/resolve/main/m2t2.pth"
```

### 6.4 运行数据生成

```bash
conda activate env_isaaclab
cd /home/gaok/coding/sage

# Headless 模式（服务器/无显示器）
python server/isaaclab/data_generation_mobile_manipulation_from_layout_parsing.py \
  --headless --enable_cameras \
  --experience isaacsim.exp.base.python.kit \
  --num_envs 1 --num_demos 1 --total_iterations_sim 200 \
  --layout_id layout_da328faf

# GUI 模式（可视化，需要显示器）
python server/isaaclab/data_generation_mobile_manipulation_from_layout_parsing.py \
  --enable_cameras \
  --experience isaacsim.exp.base.kit \
  --num_envs 1 --num_demos 1 --total_iterations_sim 200 \
  --layout_id layout_da328faf
```

> `--experience isaacsim.exp.base.python.kit` 不可省略——IsaacLab 自带的 `.kit` 文件依赖 Isaac Sim 4.x 扩展名，在 5.x 下无法解析。

### 6.5 支持的机器人

| robot_type | 底座 | 自由度 | 任务类型 |
|-----------|------|--------|---------|
| `franka` | 固定 | 7 arm + 2 gripper | 桌面抓取/放置 |
| `mobile_franka` | Omron 移动底盘 | 4 base + 7 arm + 2 gripper | 跨房间导航 + pick-and-place |

VLM 根据用户描述自动判断 robot_type：含 "mobile" → `mobile_franka`，否则 → `franka`。

### 6.6 已知问题

- **GUI 模式**：`omni.isaac.ui` 模块在 Isaac Sim 5.x 中不存在，UI 控制面板自动降级为无窗口模式，但 3D viewport 正常显示
- **显存**：`--enable_cameras` 会加载 10+ 个相机，每个 1080×1920，16 GB 显存刚好够用。如遇 OOM，需清理残留进程 `pkill -f data_generation`
- **`timeline.commit()` 卡死**：如果重新导入 URDF 后仍卡死，确认 `/tmp/IsaacLab/usd_*` 已清除，URDF 转换缓存会复用旧版本
