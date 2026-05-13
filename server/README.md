# MCP 服务端：场景生成后端

此仓库包含 MCP 场景生成管线的服务端实现。

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
conda activate sage
./client/isaac_sim_conda.sh \
  --no-window \
  omni.isaac.sim \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```
启动成功后日志中应能看到：`Isaac Sim MCP server started on localhost:8766`


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
conda activate sage
./client/isaac_sim_conda.sh \
  --no-window \
  omni.isaac.sim \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```
启动成功后日志中应能看到：`Isaac Sim MCP server started on localhost:8766`

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
conda activate sage
python client_generation_room_desc.py \
  --room_desc "A bedroom." \
  --server_paths ../server/layout_wo_robot.py
```

**机器人任务生成：**
```bash
cd /home/gaok/coding/sage/client
conda activate sage
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
