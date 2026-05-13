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
  - VLM（Qwen）、LLM（GPT）和 3D 生成模型（TRELLIS）的托管细节。
  - 运行数据增强管线的指南。

### 2. 客户端设置
**[阅读客户端文档](client/README.md)**
  - Python 环境和依赖的安装。
  - 安装和关联 NVIDIA Isaac Sim 的说明。
  - 运行场景生成、机器人任务生成和可视化的脚本。

### 使用流程
1.  **启动后端**：按照服务端 README 的说明启动 Isaac Sim MCP，并根据配置选择 Objathor 检索或 TRELLIS 生成。低显存机器默认可不启动 TRELLIS。
2.  **配置客户端**：在客户端目录中设置 `key.json` 和环境变量。
3.  **运行生成**：使用 `client/scripts/` 中的脚本来生成场景（如 `generate_from_room_desc.sh`）或机器人数据。

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
