# One-shot scene generation image

The image is a batch job, not a web service. It reads configuration from the
environment, calls the configured LLM, drives an external Isaac Sim MCP
instance, uploads the USD/thumbnail/manifest to MinIO, and exits.

Build:

```bash
docker build -f Dockerfile.scene-gen -t sage-scene-gen:isaac45 .
```

Required environment variables:

- `PROMPT`, `ISAAC_SIM_URL`, `SCENE_NAME`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`
- the project's orchestration and placement-model variables, such as
  `API_TOKEN`, `API_URL_QWEN`, `QWEN_MODEL`, `ANTHROPIC_API_KEY`,
  `ANTHROPIC_BASE_URL`, and `ANTHROPIC_MODEL`

`MINIO_OBJECT_PREFIX` is optional. `/tools/mc` must be mounted by the platform
or its path supplied with `MINIO_MC_PATH`.

Objathor embedding retrieval uses CLIP and `all-mpnet-base-v2`. For an offline
job, mount the SentenceTransformer repository at
`/models/all-mpnet-base-v2` (auto-detected), and mount a populated Hugging Face
cache at `/root/.cache/huggingface`. `SAGE_SBERT_MODEL` can point to a different
mounted location. Without these mounts the libraries download their weights on
first use.

The Isaac extension consumes absolute JSON and asset paths. Therefore the
Objathor directory and generated-results directory must be mounted at the same
absolute paths in both the job container and the external Isaac host. The
defaults are `/shared/objathor` and `/shared/results`; override them with
`SAGE_OBJATHOR_ROOT` and `SAGE_RESULTS_DIR` when the host uses another shared
path.

Example (Isaac and the container run on the same Linux host):

```bash
docker run --rm --network host \
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
  -e MINIO_ENDPOINT -e MINIO_ACCESS_KEY -e MINIO_SECRET_KEY \
  -e MINIO_BUCKET -e MINIO_OBJECT_PREFIX \
  -e API_TOKEN -e API_URL_QWEN -e QWEN_MODEL \
  -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_MODEL \
  sage-scene-gen:isaac45
```

No port is exposed and no GPU device is requested by the container.
