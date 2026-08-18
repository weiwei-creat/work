# SAGE 场景生成 Docker 部署文档

## 1. 交付文件

以下文件和资源当前位于单卡开发机 `172.16.41.23`：

- Docker 镜像文件：`/data/gaok/sage/sage-scene-gen-isaac45.tar`
- 镜像文件大小：1,823,804,416 字节（约 1.70 GiB）
- 镜像标签：`sage-scene-gen:isaac45-reproduced`
- 23 号开发机当前镜像 ID：`sha256:b5a574928799c75734ddd42f944d6add359ab1f447e8f9ea695944a82fb87822`
- 镜像文件 SHA-256：`898eaf01682115fb8661cd342355e30036bce574b40012ed886dfa55484b1404`
- 项目目录：`/data/gaok/sage`
- Objathor 资源目录：`/data/gaok/sage/objathor`，约 45 GB
- 场景结果目录：`/data/gaok/sage/runtime/results`
- MinIO 客户端：`/data/gaok/sage/runtime/tools/mc`
- SentenceTransformer 模型：`/home/ubuntu/models/all-mpnet-base-v2`
- Hugging Face 模型缓存：`/home/ubuntu/.cache/huggingface`
- Isaac Sim：`/home/ubuntu/isaac`，版本 4.5.0
- Conda 环境：`/home/ubuntu/miniconda3/envs/sage`

校验镜像文件：

```bash
cd /data/gaok/sage
sha256sum sage-scene-gen-isaac45.tar
```

输出应为：

```text
898eaf01682115fb8661cd342355e30036bce574b40012ed886dfa55484b1404  sage-scene-gen-isaac45.tar
```

> 镜像文件当前属于 `ubuntu` 用户。如果其他用户无法读取，请通过 `ubuntu` 用户执行部署，
> 或由管理员调整文件读取权限。

## 2. 镜像职责与运行方式

该镜像是一次性场景生成任务，不是常驻 HTTP 服务。完整流程如下：

```text
启动任务容器
  → 读取环境变量
  → 调用 LLM 编排场景
  → 连接容器外部的 Isaac Sim MCP
  → 生成 USD 和缩略图
  → 上传标准入口文件和完整场景产物目录至 MinIO
  → 容器退出
```

部署约束：

- 任务容器不暴露 HTTP 端口，不需要配置 `-p`。
- 任务容器不申请 GPU，不需要配置 `--gpus`。
- Isaac Sim 4.5 在容器外部常驻，并使用宿主机 GPU。
- MinIO 客户端由平台挂载至容器内的 `/tools/mc`。
- 每次场景生成启动一个新容器；完成后容器以退出码 `0` 结束。
- Objathor 和结果目录必须同时对任务容器与外部 Isaac Sim 可见，而且绝对路径必须一致。

镜像入口为：

```text
python /app/docker/scene_entrypoint.py
```

Dockerfile 中的入口定义为：

```dockerfile
ENTRYPOINT ["python", "/app/docker/scene_entrypoint.py"]
```

## 3. 部署前提

任务执行前应确认：

1. 宿主机已经安装 Docker。
2. 外部 Isaac Sim 4.5 可以正常使用 GPU 启动。
3. `sage` Conda 环境已经配置完成。
4. Objathor、SentenceTransformer 和 Hugging Face 缓存目录存在。
5. MinIO 服务可从任务容器访问，目标存储桶已经创建。
6. 平台已经准备好 MinIO `mc` 二进制文件。
7. 已准备可用的 LLM API 地址、模型名和访问密钥。

LLM API 由镜像使用者在任务启动时提供。镜像本身不包含开发机的 `.env`、固定 API 地址或
访问密钥，23 号开发机上用于验收的 API 停用后不会影响其他用户使用自己的 API。

检查关键文件：

```bash
test -x /home/ubuntu/isaac/kit/kit
test -x /home/ubuntu/miniconda3/envs/sage/bin/python
test -d /data/gaok/sage/objathor
test -d /home/ubuntu/models/all-mpnet-base-v2
test -x /data/gaok/sage/runtime/tools/mc
```

## 4. 导入镜像

```bash
cd /data/gaok/sage
docker load -i sage-scene-gen-isaac45.tar
```

确认镜像已经导入：

```bash
docker image inspect sage-scene-gen:isaac45-reproduced \
  --format '镜像={{.Id}} 入口={{json .Config.Entrypoint}}'

docker image inspect sage-scene-gen:isaac45-reproduced \
  --format '{{json .Config}}'
```

预期可以看到 Python 入口，并且镜像配置中不存在 `ExposedPorts`。

## 5. 启动外部 Isaac Sim MCP

任务镜像不会在容器内启动 Isaac Sim。必须先在 GPU 宿主机上启动 Isaac Sim 4.5 MCP。

先检查 MCP 是否已经运行：

```bash
ss -ltnp | grep ':11323'
```

如果已经显示 `127.0.0.1:11323` 正在监听，不要重复启动。若尚未运行，可执行：

```bash
cd /data/gaok/sage
mkdir -p runtime/results runtime/logs

# 避免 Isaac 扩展较多时耗尽进程文件句柄
ulimit -n 65536

nohup env \
  CONDA_PYTHON=/home/ubuntu/miniconda3/envs/sage/bin/python \
  ISAAC_SIM_PATH=/home/ubuntu/isaac \
  SAGE_OBJATHOR_ROOT=/data/gaok/sage/objathor \
  SAGE_RESULTS_DIR=/data/gaok/sage/runtime/results \
  SAGE_ISAAC_BIND_HOST=127.0.0.1 \
  SAGE_ISAAC_MCP_PORT=11323 \
  ./client/isaac_sim_conda.sh \
    --experience isaacsim.exp.base.kit \
    --no-window \
    --ext-folder /data/gaok/sage/server/isaacsim \
    --enable isaac.sim.mcp_extension \
  > runtime/logs/isaac45.log 2>&1 &

echo $! > runtime/isaac45.pid
```

首次启动可能需要几分钟。查看启动日志：

```bash
tail -f /data/gaok/sage/runtime/logs/isaac45.log
```

确认 MCP 端口已经就绪：

```bash
ss -ltnp | grep ':11323'
timeout 3 bash -c '</dev/tcp/127.0.0.1/11323' && echo 'Isaac MCP 已就绪'
```

> 如果任务容器与 Isaac Sim 不在同一台主机，应把 `SAGE_ISAAC_BIND_HOST` 设置为允许任务节点
> 访问的地址，并放通 TCP 11323。此端口是 Isaac MCP 端口，不是镜像提供的 HTTP 服务端口。

## 6. 准备任务环境变量

推荐使用权限为 `600` 的环境变量文件，避免把密钥直接写入命令历史。创建文件：

```bash
mkdir -p /data/gaok/sage/runtime
touch /data/gaok/sage/runtime/scene-job.env
chmod 600 /data/gaok/sage/runtime/scene-job.env
```

编辑 `/data/gaok/sage/runtime/scene-job.env`：

```dotenv
# 本次任务
PROMPT=一间紧凑的现代卧室，包含床、衣柜、书桌、椅子和床头柜
SCENE_NAME=modern-bedroom-001
ISAAC_SIM_URL=tcp://127.0.0.1:11323

# MinIO
MINIO_ENDPOINT=http://<MinIO地址>:<端口>
MINIO_ACCESS_KEY=<MinIO访问密钥>
MINIO_SECRET_KEY=<MinIO私有密钥>
MINIO_BUCKET=<存储桶名称>
MINIO_OBJECT_PREFIX=scenes/modern-bedroom-001

# 场景编排模型，接口需要兼容 OpenAI Chat Completions
API_TOKEN=<编排模型访问密钥>
API_URL_QWEN=<编排模型接口地址>
QWEN_MODEL=<编排模型名称>

# 布局、材质和 critic 模型，接口需要兼容 Anthropic SDK
ANTHROPIC_API_KEY=<模型访问密钥>
ANTHROPIC_BASE_URL=<Anthropic兼容接口地址>
ANTHROPIC_MODEL=<模型名称>

# 当前 DeepSeek 文本模型不接收图片，使用它时应关闭语义 VLM critic
SEMANTIC_CRITIC_ENABLED=false
PHYSICS_CRITIC_ENABLED=true

# 可选性能参数
SAGE_MAX_TOOL_CALLS=40
SAGE_PREVIEW_RESOLUTION=384
SAGE_PREVIEW_VIEWS=3
SAGE_FULL_PREVIEW_VIEWS=4
```

两组 LLM 变量可以使用同一个模型、密钥和服务地址，但服务端必须同时提供 OpenAI Chat
Completions 兼容接口和 Anthropic SDK/API 兼容接口。若服务只实现一种协议，应分别提供接口
或在服务前增加协议适配层。

变量说明：

| 变量 | 必需 | 说明 |
|---|---:|---|
| `PROMPT` | 是 | 场景自然语言描述 |
| `SCENE_NAME` | 是 | 输出场景名，只能使用字母、数字、点、下划线和连字符 |
| `ISAAC_SIM_URL` | 是 | 外部 Isaac MCP 地址，默认端口为 11323 |
| `MINIO_ENDPOINT` | 是 | MinIO 服务地址，必须包含 `http://` 或 `https://` |
| `MINIO_ACCESS_KEY` | 是 | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | 是 | MinIO 私有密钥 |
| `MINIO_BUCKET` | 是 | 已存在的目标存储桶 |
| `MINIO_OBJECT_PREFIX` | 否 | MinIO 对象路径前缀 |
| `API_TOKEN` | 是 | 场景编排模型访问密钥 |
| `API_URL_QWEN` | 是 | OpenAI 兼容的编排模型接口地址 |
| `QWEN_MODEL` | 是 | 编排模型名称 |
| `ANTHROPIC_API_KEY` | 是 | 布局及 critic 模型访问密钥 |
| `ANTHROPIC_BASE_URL` | 视接口而定 | Anthropic 或 Anthropic 兼容接口地址 |
| `ANTHROPIC_MODEL` | 是 | 布局及 critic 模型名称 |

## 7. 启动一次场景生成任务

确保共享结果目录可由任务容器和外部 Isaac Sim 写入：

```bash
mkdir -p /data/gaok/sage/runtime/results
chmod a+rwx /data/gaok/sage/runtime/results
```

前台运行方式如下。前台运行便于直接观察日志和取得退出码：

```bash
cd /data/gaok/sage

docker run --rm \
  --name sage-scene-modern-bedroom-001 \
  --network host \
  --env-file /path/to/your/scene-job.env
  -e SAGE_OBJATHOR_ROOT=/data/gaok/sage/objathor \
  -e SAGE_RESULTS_DIR=/data/gaok/sage/runtime/results \
  -v /data/gaok/sage/objathor:/data/gaok/sage/objathor:ro \
  -v /data/gaok/sage/runtime/results:/data/gaok/sage/runtime/results \
  -v /home/ubuntu/models/all-mpnet-base-v2:/models/all-mpnet-base-v2:ro \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /data/gaok/sage/runtime/tools/mc:/tools/mc:ro \
  sage-scene-gen:isaac45-reproduced
```

注意：

- 不要添加 `-p`，镜像没有 HTTP 服务。
- 不要添加 `--gpus`，镜像使用 CPU，渲染由外部 Isaac Sim 完成。
- 同机部署使用 `--network host`，这样容器可以通过 `127.0.0.1:11323` 连接 Isaac MCP。
- 不要把真实密钥写入部署文档或提交到 Git。

如果平台需要后台执行，可使用：

```bash
docker run -d \
  --name sage-scene-job-001 \
  --network host \
  --env-file /data/gaok/sage/runtime/scene-job.env \
  -e SAGE_OBJATHOR_ROOT=/data/gaok/sage/objathor \
  -e SAGE_RESULTS_DIR=/data/gaok/sage/runtime/results \
  -v /data/gaok/sage/objathor:/data/gaok/sage/objathor:ro \
  -v /data/gaok/sage/runtime/results:/data/gaok/sage/runtime/results \
  -v /home/ubuntu/models/all-mpnet-base-v2:/models/all-mpnet-base-v2:ro \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /data/gaok/sage/runtime/tools/mc:/tools/mc:ro \
  sage-scene-gen:isaac45-reproduced

docker logs -f sage-scene-job-001
docker wait sage-scene-job-001
docker inspect sage-scene-job-001 --format '退出码={{.State.ExitCode}} 状态={{.State.Status}}'
docker rm sage-scene-job-001
```

`docker wait` 输出 `0` 表示任务成功；非 `0` 表示任务失败，应查看容器日志。

## 8. 验证任务结果

日志中出现以下内容表示上传完成：

```text
[scene-gen] artifacts uploaded
```

MinIO 中会生成三个标准入口对象，并递归保存本次 `layout_id` 的完整结果目录：

```text
<MINIO_OBJECT_PREFIX>/<SCENE_NAME>.usd
<MINIO_OBJECT_PREFIX>/thumb.png
<MINIO_OBJECT_PREFIX>/manifest.json
<MINIO_OBJECT_PREFIX>/<layout_id>/
├── <layout_id>.json
├── <layout_id>.usd
├── <layout_id>.usdz
├── <layout_id>_view.usd
├── <layout_id>_usd_collection/
├── materials/
├── objaverse/
├── preview/
├── room_<id>.json
├── room_<id>.usd
└── room_<id>.usdz
```

完整目录会保留所有子目录及文件的相对路径，确保 USD/USDZ、材质、网格、纹理和预览之间的
引用关系不因上传而改变。`.placements-*` 等隐藏临时文件及 `__pycache__` 不会上传。

主场景、房间场景和 USD collection 遵循以下交付约定：

| 项目 | 固定值/行为 |
|---|---|
| 舞台朝向 | `upAxis=Z` |
| 比例尺 | `metersPerUnit=1.0`，一单位等于一米 |
| 根节点 | 不添加补偿旋转或缩放 |
| 房顶 | 不导出 roof/ceiling Prim 或 collection 文件 |
| 场景照明 | USD 自带 DomeLight 和 DistantLight，不依赖 Isaac 视口默认灯 |

如果收到的 USD 是 `Y-up` 或 `metersPerUnit=0.01`，说明仍在使用旧版产物；米制 Isaac
资产加入该旧舞台后可能显得约100倍过大，且场景可能侧倒。应重新导入最新镜像并重新生成。

其中 `manifest.json` 的结构如下：

```json
{
  "scene_name": "modern-bedroom-001",
  "layout_id": "layout_xxxxxxxx",
  "usd": "modern-bedroom-001.usd",
  "thumbnail": "thumb.png",
  "artifact_root": "layout_xxxxxxxx/",
  "artifact_count": 83,
  "artifacts": [
    "layout_xxxxxxxx.json",
    "layout_xxxxxxxx.usd",
    "layout_xxxxxxxx.usdz",
    "materials/room_xxxxxxxx_floor.png"
  ]
}
```

`artifact_count` 取决于场景复杂度，示例中的 `83` 不是固定值；`artifacts` 会列出该任务实际
上传的全部相对路径。

宿主机共享结果目录中也会保留原始生成结果：

```bash
find /data/gaok/sage/runtime/results -maxdepth 3 \
  -type f \( -name '*.usd' -o -name '*.png' -o -name '*.json' \) | sort
```

## 9. 平台接入要求

平台应将该镜像按批处理任务或 Kubernetes Job 接入，而不是创建 HTTP Service：

- 每个生成请求创建一个独立任务。
- 通过环境变量或 Secret 注入 LLM、MinIO 和任务参数。
- 使用 `restartPolicy: Never`，以容器退出码判断成功或失败。
- 不创建 Service、Ingress 或 HTTP 探针。
- 不申请 GPU 资源。
- 通过共享卷提供 Objathor 和结果目录。
- 通过 initContainer 将 MinIO `mc` 放到共享卷的 `/tools/mc`。
- 外部 Isaac Sim MCP 应先启动并保持可达。
- 建议任务超时时间至少设置为 30～60 分钟，复杂场景可适当增加。

如果任务容器和 Isaac Sim 位于不同节点，共享存储必须在两侧挂载为相同绝对路径。例如：

```text
/shared/objathor
/shared/results
```

然后相应设置：

```dotenv
SAGE_OBJATHOR_ROOT=/shared/objathor
SAGE_RESULTS_DIR=/shared/results
ISAAC_SIM_URL=tcp://<Isaac宿主机地址>:11323
```

## 10. 停止 Isaac Sim MCP

确认没有其他场景任务正在使用 Isaac Sim 后执行：

```bash
kill "$(cat /data/gaok/sage/runtime/isaac45.pid)"
rm -f /data/gaok/sage/runtime/isaac45.pid
```

如果 PID 文件不存在，可先查找进程：

```bash
pgrep -af '/home/ubuntu/isaac/kit/kit'
```

## 11. 常见问题

### 11.1 `missing required environment variable`

环境变量缺失。检查 `scene-job.env` 是否包含日志中指出的变量，以及是否正确传入
`--env-file`。

### 11.2 `external Isaac MCP` 连接失败或 `Connection refused`

检查 Isaac Sim 是否正在运行以及 11323 端口是否监听：

```bash
ss -ltnp | grep ':11323'
tail -n 200 /data/gaok/sage/runtime/logs/isaac45.log
```

### 11.3 `Objathor data not found`

检查 `SAGE_OBJATHOR_ROOT` 和 `-v` 的容器内路径是否完全一致，并确认外部 Isaac Sim
也能使用同一绝对路径读取资源。

### 11.4 `MinIO client not found: /tools/mc`

检查 MinIO 客户端是否存在且可执行，并确认已经挂载：

```bash
ls -l /data/gaok/sage/runtime/tools/mc
```

### 11.5 MinIO 上传失败

检查 Endpoint、密钥、存储桶、网络连通性和写入权限。`MINIO_ENDPOINT` 必须包含协议，
例如 `http://minio.example.internal:9000`。

### 11.6 模型权重下载失败

离线部署必须同时挂载：

```text
/home/ubuntu/models/all-mpnet-base-v2 → /models/all-mpnet-base-v2
/home/ubuntu/.cache/huggingface       → /root/.cache/huggingface
```

### 11.7 结果目录权限不足

任务容器与 Isaac Sim 可能使用不同 UID。确认两者均可写入共享结果目录，并避免将结果目录
以只读方式挂载。

## 12. 已完成的验证

当前交付 tar 对应的镜像已于 2026-07-24 在 23 号开发机完成一次完整端到端部署链路测试：

- 容器成功连接外部 Isaac Sim 4.5 MCP。
- 成功生成 USD 场景和 384×384 PNG 缩略图。
- 成功上传 USD、缩略图和清单至 MinIO。
- 容器最终退出码为 `0`。
- 容器没有暴露端口。
- 容器没有申请 GPU 设备。

本次验收任务名称为 `repair-final-20260724`，MinIO 中实际生成：

```text
validation/final-20260724/repair-final-20260724.usd  124 KiB
validation/final-20260724/thumb.png                   142 KiB
validation/final-20260724/manifest.json               143 B
```

上述结果证明“读取运行时环境变量 → 调用外部 LLM API → 连接 Isaac Sim MCP → 导出场景
→ 上传 MinIO → exit 0”部署链路可用。它不等同于对所有提示词的场景语义准确率作出保证；
本次日志中桌面物体的表面放置未达到提示词预期，属于后续算法质量优化项，不影响镜像部署
链路验收结论。

2026-08-03 对完整结果目录上传进行了追加验证：源目录
`layout_1394ed1a` 共 83 个文件、55,769,816 字节，MinIO 对应前缀中成功生成 83 个对象，
顶层 USD/USDZ/JSON 及 `layout_1394ed1a_usd_collection/`、`materials/`、`objaverse/`、
`preview/` 均已保留。

2026-08-18 使用 Isaac Sim 4.5 对 `layout_1394ed1a` 进行了 USD 交付修复和磁盘文件回读验证：

- Isaac 通过 `open_stage` 直接打开 `/data/gaok/sage/runtime/results/layout_1394ed1a/layout_1394ed1a.usd`；
- 检测结果为 `upAxis=Z`、`metersPerUnit=1.0`，根节点无补偿旋转；
- 实际网格边界约为 `3.60 × 4.25 × 2.80 m`；
- USD 内含两盏场景灯，ceiling/roof Prim 数为 `0`；
- 主 USD、主 USDZ、房间 USD、房间 USDZ 均通过同一检查；
- USD collection 共30个文件、14个 USD，ceiling文件数为 `0`；
- Isaac 从已打开的磁盘 USD 渲染结果保存为
  `preview/layout_1394ed1a_opened_in_isaac_fixed.png`。
- 最终镜像内重试上传会先清理该 layout 的 MinIO 对象前缀；本次筛选后本地产物
  85 个、48,482,229 字节，MinIO 对应前缀为 85 个对象，隐藏临时文件为 `0`。
