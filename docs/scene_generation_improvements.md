# SAGE 场景生成改进（对照 SceneSmith）

本文件记录参考 **SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes**
(arXiv:2602.09153) 对 SAGE 场景生成 prompt / 流程做的改进，区分**已落地**、
**部分落地**、**待办（需额外资产/重启）**。

## 已落地（代码已改，远程编译通过）

### 1. 放开工具调用上限 + 历史摘要 —— 提升物体密度
- 三个客户端 `max_tool_calls` 从硬编码 15 → 默认 40，可用 `SAGE_MAX_TOOL_CALLS` 配置。
- 旧的工具结果（大 JSON，上下文膨胀主因）按 `SAGE_FULL_TOOL_RESULT_WINDOW`(默认6)
  保留最近若干条全文，更早的截断到 `SAGE_TOOL_RESULT_TRUNCATE_CHARS`(默认600) 字符头部。
  确定性截断、无额外 LLM 调用，让高 `max_tool_calls` 不撑爆上下文。
- 文件：`client/client_generation_robot_task.py`、`client_generation_room_desc.py`、`client_generation.py`

### 2. 检查点 + 回滚 —— 防止放置把场景改差（仿 orchestrator rollback）
- `place_objects_in_room` 放置前对房间物体打快照。
- 语义 critic 现在透传 `overall_room_rating`(excellent/good/fair/poor) 与 `highest_issue_priority`。
- 放置后若评级**低于上一次**（regression），回滚到快照并在返回里给 `rollback` 字段，
  指示设计者换更谨慎的方案。
- **三信号**：(A) 语义最严重问题严重度 `highest_issue_priority`(0-10) 较上次新增
  ≥ `SAGE_SEMANTIC_ROLLBACK_DELTA`(默认3) —— **总有值、最可靠**；(A') 分类评级 overall_room_rating
  下降（模型可能漏填，兜底）；(B) 物理不稳定物体数较上次新增 ≥ `SAGE_PHYSICS_ROLLBACK_DELTA`(默认3)。
- 开关：`SAGE_ENABLE_ROLLBACK`(默认 true)。文件：`server/layout.py`
- **已修（本机实测发现并解决）**：语义 critic 渲染曾失败
  （`Isaac/RTX preview rendering failed and CPU fallback is disabled`）。
  根因不是环境选择：`sage` 与 `sage5080` 的 torch 都是 2.5.1+cu124，**都不支持 RTX 5080 的
  sm_120（Blackwell）算力**（CUDA "no kernel image"），GPU/nvdiffrast 渲染在两个环境下都跑不了
  （sage5080 更糟，nvdiffrast .so ABI 不匹配，import 即崩）。
  `server/room_render.py` 的 CPU 回退本已完整实现但被 `_CPU_RENDER_FALLBACK_ENABLED=False` 硬关。
  **修法**：改为环境变量控制、**默认开启** CPU 回退（`SAGE_ENABLE_CPU_RENDER_FALLBACK`，默认1）。
  实测对 layout_ca161c55：top_orthogonal/four_edges/four_top 三类渲染全部出图。
  → 语义 critic、#2 语义信号、#5 视觉接地恢复可用。
  彻底用 GPU 渲染需把 torch 升到支持 sm_120 的版本（torch≥2.7 / cu128），风险高，暂用 CPU 回退。
- **连带修复的潜伏 bug**：渲染恢复后暴露出 `room_semantic_critic` 里 `room_description` 未定义
  （`NameError`，之前被渲染早退掩盖）。已补 `room_description = get_room_description(room)`。
  实测语义 critic 现完整跑通：渲染→Claude→返回 next_step + highest_issue_priority(实测=8)。

### 4. 量化评测 harness —— 让所有改进可度量
- 新脚本 `server/eval_scene_metrics.py`，纯几何 + occupancy，**无需 Isaac**。
- 指标：CNT(物体数)、COL(地面家具碰撞率，真实信号；桌面聚集另算且信息性)、
  OOB(越界)、NAV(最大连通可走面积占比)、ACC(地面物体可达率)、STB*(支撑几何代理)。
- 用法：`conda activate sage && python server/eval_scene_metrics.py --layout_id <id>`
- 已在 layout_d553cff2 实测：CNT=31, COL=0%, OOB=0%, NAV=1.00, ACC=100%, STB*=100%。

### 6. 分面密集放置 + 9. 未填充支撑面结构化输出
- `place_objects_in_room` 返回新增 `underpopulated_surfaces`：列出 items < `SAGE_MIN_ITEMS_PER_SURFACE`(默认2)
  的支撑面（桌/书架/柜台/床头柜等），含 `supporter_id` 与当前/所需数量。
- 客户端 prompt 新增 SOURCE 4，指示设计者把该列表当 work-list，对每个 supporter_id
  做分组密集放置，直到清空 —— 把 SceneSmith 的"支撑面层级 + 组合放置"意图落到行为上
  （未引入新工具，避免 agent 混淆；place_objects 本就支持自然语言多物体分组放置）。
- prompt 同时新增 rollback 提示：收到 `rollback` 字段时换方案、勿重复。

## 部分落地 / 已存在

### 5. 设计者视觉接地
- SceneSmith 让 designer 每步看渲染图。SAGE **已有**：语义 critic 渲染 4 张俯视 + 透视图
  交给 Claude 分析，把视觉派生的建议（文本）回灌设计者 —— 即 SAGE 不处于论文
  "NoObserveScene"(无视觉) 的退化档位。
- **完整版**（把渲染图直接喂给放置决策的 Qwen3-VL）需要：place_objects 通过 MCP 返回
  Image + 客户端把工具结果图片拼进下一轮消息。属于较大改动（MCP 图片管线 + token 成本），
  暂缓。集成点：`place_objects_in_room` 末尾已渲染但丢弃的 viz（layout.py ~3110 行附近）。

## 待办（需额外资产 / 需重启 kit，未盲改）

### 3. 物理沉降回写
- kit 的 `simulate_the_scene` handler **已计算**沉降后位姿 `final_position/orientation`，
  但 return 注释掉了 `simulation_result`，只回传稳定性标志（`extension.py` ~1880 行）。
- SAGE **已有**物理 critic（`room_physics_critic` 跑 simulate_the_scene 反馈不稳定 + 移除超天花板物体）。
- 要做 SceneSmith 式"沉降回写"需两处改动：
  1. `extension.py::simulate_the_scene` 增加 `include_transforms` 参数，返回 `final_position/orientation`；
  2. `server/layout.py` 加 `settle_and_writeback(room_id)`：调 simulate → 读沉降位姿 → 写回 layout → 重导出，
     gated `SAGE_ENABLE_SETTLE`。
- **未实施原因**：第 1 步需重启正在运行的 MCP kit 才能生效，且需完整生成跑一遍验证，
  无法在不影响你生产 kit 的前提下盲改验证。与现有物理 critic 高度重叠，优先级低。

### 7. 关节物体（可开合家具）
- SceneSmith 用 ArtVIP 库 + CLIP 两段检索放可开合抽屉/柜门。
- SAGE 有零散关节支持（门 joint、articulation 实验），但**缺关节资产库**。
- 集成点：资产检索处增加关节资产来源 + 放置时标注 articulation。需先有资产库，暂缓。

## 资产污染提示（公平评测）
- 做策略**泛化评测**的场景应走 TRELLIS 按需生成（`SAGE_ENABLE_TRELLIS_GENERATION=1`,
  `SAGE_OBJECT_SOURCE=generation`），避免 Objathor 检索资产与训练数据重叠造成评测不公。

## 新增环境变量一览
| 变量 | 默认 | 作用 |
|------|------|------|
| `SAGE_MAX_TOOL_CALLS` | 40 | 单次生成最大工具调用数（密度上限） |
| `SAGE_FULL_TOOL_RESULT_WINDOW` | 6 | 保留全文的最近工具结果条数 |
| `SAGE_TOOL_RESULT_TRUNCATE_CHARS` | 600 | 旧工具结果截断长度 |
| `SAGE_ENABLE_ROLLBACK` | true | 放置变差时回滚 |
| `SAGE_MIN_ITEMS_PER_SURFACE` | 2 | 支撑面"未填充"判定阈值 |
