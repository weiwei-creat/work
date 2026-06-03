# MCP Client: Scene Generation Interface

This repository contains the client-side implementation for controlling the Scene Generation pipeline.

## 1. Installation & Setup

### 1.1 Python Environment
Install the `sage` Python environment using the provided configuration at `./environment.yml`.

### 1.2 Material Generation Engine
1.  **Matfuse:** Follow the installation guidance at `../matfuse-sd`.
2.  **Configuration:** Update the Matfuse directory path in `./constant.py`.
*   *Note:* Flux is also supported. See `./start_flux_server.sh` (Requires `HF_TOKEN`).

### 1.3 Isaac Sim MCP Service
SAGE is currently verified with Isaac Sim 5.1. Set `ISAAC_SIM_PATH` if Isaac Sim is not installed in the default location used by `client/isaac_sim_conda.sh`.

The MCP extension is loaded from `server/isaacsim` at launch time, so no symlink into the Isaac Sim install directory is required.

Start the service from the repo root:
```bash
cd /home/gaok/coding/sage
conda activate sage5080
./client/isaac_sim_conda.sh \
  --no-window \
  --experience isaacsim.exp.base.kit \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```

Verify the connection: the Isaac log should contain `Isaac Sim MCP server started on localhost:11323` or the port assigned from `SLURM_JOB_ID`.

For Isaac Sim 5.x source builds, keep `--experience isaacsim.exp.base.kit`. The wrapper launches `kit/kit` directly so this experience is honored instead of falling back to the Full app.

### 1.4 VLM / LLM Configuration
The current default flow uses hosted API endpoints configured in `.env` and `client/key.json`; no local VLM service is required unless you explicitly choose self-hosting. See `../server/README.md` for the current DashScope / DeepSeek and optional self-hosted Qwen3-VL setup.

### 1.5 Configuration
Fill in the API token and URL in `./key.json`.

## 2. Scene Generation Usage

### 2.1 Start Isaac Sim (Background Service)
Start the Isaac Sim MCP server from the repo root using the command in section 1.3 before running generation scripts.

### 2.2 Generation Commands
Once Isaac Sim is running, use the following scripts to generate scenes.

**I. Generate from Single Room Descriptions**
```bash
./scripts/generate_from_room_desc.sh
```

**II. Generate from Robot Task Descriptions**
```bash
./scripts/generate_from_robot_task.sh
```

**III. Multi-Room Generation**
```bash
./scripts/generate_multi_rooms.sh
```

*   **Image Conditioning:** All scripts support image conditioning by appending the image path as an input argument. Prompts might need revisions.

### 2.3 Exports scenes to GLB, USD, and for Rendering
First, you need to pack up the scenes with 
```
cd ../server
python pack_scene_to_zip.py --layout_id [LAYOUT_ID] --upload_name [LAYOUT_ID]
```

Then unzip and use the kits in `https://huggingface.co/datasets/nvidia/SAGE-10k/tree/main/kits` to export to glb, usd, and rendering.


## 3. Scene Layout-Level Augmentation

For layout-level augmentation (where task-related objects are maintained while the rest of the layout is re-generated):

**Prerequisite:** A base layout is required.

**Run Augmentation:**
```bash
./scripts/generate_scene_augmentation.sh
```
