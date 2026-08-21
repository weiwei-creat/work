# 一次性场景生成镜像

该镜像用于执行一次性批处理任务，而不是提供 Web 服务。容器启动后会从环境变量中读取配置，
调用指定的大语言模型（LLM），驱动外部 Isaac Sim MCP 实例生成场景，将便于平台消费的 USD、
缩略图和清单，以及保留原始目录结构的完整场景产物上传至 MinIO，然后自动退出。

完整的镜像交付信息、Isaac Sim MCP 启动方法、任务运行命令和故障排查说明，请参阅
[SAGE 场景生成 Docker 部署文档](./SAGE场景生成Docker部署.md)。

## 构建镜像

```bash
docker build -f Dockerfile.scene-gen -t sage-scene-gen:isaac45-hierarchy-final-20260821 .
```

## 必需的环境变量

- `PROMPT`, `ISAAC_SIM_URL`, `SCENE_NAME`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`
- 项目编排及物体放置模型所需的变量，例如 `API_TOKEN`、`API_URL_QWEN`、
  `QWEN_MODEL`、`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL` 和
  `ANTHROPIC_MODEL`

镜像不包含固定的 LLM 地址、模型名或访问密钥，也不依赖开发机上的 `.env`。镜像使用者必须在
每次启动任务时通过 `--env-file`、Docker Secret 或 `-e` 注入自己的 LLM 配置。两组变量可以
指向同一个模型服务，但该服务必须同时兼容 OpenAI Chat Completions 和 Anthropic SDK/API；
如果只兼容其中一种协议，需要分别提供两个接口或增加协议适配层。

`MINIO_OBJECT_PREFIX` 为可选变量。平台必须将 MinIO 客户端挂载到 `/tools/mc`，
也可以通过 `MINIO_MC_PATH` 指定其他客户端路径。

Objathor 嵌入检索使用 CLIP 和 `all-mpnet-base-v2`。如果需要离线运行，请将
SentenceTransformer 模型仓库挂载到 `/models/all-mpnet-base-v2`（程序会自动检测），
并将已经包含所需模型的 Hugging Face 缓存挂载到 `/root/.cache/huggingface`。
也可以通过 `SAGE_SBERT_MODEL` 指向其他已挂载的模型位置。如果没有挂载这些目录，
相关库会在首次使用时联网下载模型权重。

Isaac 扩展使用 JSON 文件和资源文件的绝对路径。因此，Objathor 目录和场景生成结果目录必须
以相同的绝对路径同时挂载到任务容器与外部 Isaac 宿主机。默认路径分别为
`/shared/objathor` 和 `/shared/results`。如果宿主机使用其他共享路径，请通过
`SAGE_OBJATHOR_ROOT` 和 `SAGE_RESULTS_DIR` 覆盖默认值。

## 运行示例

以下示例假定 Isaac Sim 和任务容器运行在同一台 Linux 宿主机上：

```bash
chmod 600 /path/to/scene-job.env

docker run --rm --network host \
  --env-file /path/to/scene-job.env \
  -v /data/gaok/sage/objathor:/data/gaok/sage/objathor:ro \
  -v /data/gaok/sage/runtime/results:/data/gaok/sage/runtime/results \
  -v /home/ubuntu/models/all-mpnet-base-v2:/models/all-mpnet-base-v2:ro \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /opt/tools/mc:/tools/mc:ro \
  -e PROMPT='A compact modern bedroom' \
  -e ISAAC_SIM_URL='tcp://127.0.0.1:11323' \
  -e SCENE_NAME='modern-bedroom' \
  -e SAGE_OBJATHOR_ROOT='/data/gaok/sage/objathor' \
  -e SAGE_RESULTS_DIR='/data/gaok/sage/runtime/results' \
  sage-scene-gen:isaac45-hierarchy-final-20260821
```

`scene-job.env` 应包含上文列出的任务、LLM 和 MinIO 变量，但不得提交到 Git 或随镜像分发。
该容器不会暴露任何端口，也不会申请 GPU 设备。GPU 仅由容器外部运行的 Isaac Sim 使用。

## USD 交付约定

交付的主 USD、房间 USD 和对应 USDZ 统一采用 Isaac Sim 原生约定：

- `upAxis=Z`，不在根节点附加补偿旋转；
- `metersPerUnit=1.0`，几何尺寸以米为单位；
- 不包含 roof/ceiling Prim，打开后直接显示室内；
- USD 在 `/World/Room/Lights` 内置 `DomeLight` 和 `DistantLight`，关闭
  Isaac 视口默认灯后场景仍有照明；
- USD collection 保持同样的 Z-up/米制元数据，并且不导出 ceiling 文件。

主 USD 的场景树固定为：

```text
/World
├── /Room                         # 固定环境，sage:editable=false
│   ├── /Floor
│   ├── /Walls
│   ├── /Windows
│   ├── /Doors               # 门框、门板和 Hinge 属于 Room
│   └── /Lights
└── /Objects                      # 可编辑物件
    └── /<Type>_<id>              # 独立 Xform，含位置和旋转
        ├── /Geometry              # 局部坐标 Mesh
        └── /Material
```

`/World/Objects` 下不再使用已烘焙世界坐标的家具 Mesh；每件物体的
`position`/`rotation` 写在自身 Xform 上，因此可在 Isaac Sim 中独立选中、平移和旋转。

不要把旧版本中声明为 `Y-up`、`metersPerUnit=0.01` 的原始 USD 作为交付物。该旧元数据会让
Isaac 米制资产表现出旋转异常和约 100 倍的比例差异。

上传完成后，MinIO 的任务前缀下包含三个标准入口文件和一个完整结果目录：

```text
<SCENE_NAME>.usd
thumb.png
manifest.json
<layout_id>/
  ├── <layout_id>.json/.usd/.usdz
  ├── <layout_id>_view.usd
  ├── <layout_id>_usd_collection/
  ├── materials/
  ├── objaverse/
  ├── preview/
  └── room_*.json/.usd/.usdz
```

`<SCENE_NAME>.usd` 固定复制主文件 `<layout_id>.usd`，不再优先使用仅供预览的
`<layout_id>_view.usd`，因此 MinIO 标准入口保留上述 Room/Objects 层级。
