# SAGE Isaac Sim MCP Extension

此目录包含 SAGE 使用的 Isaac Sim MCP 扩展与轻量 MCP 客户端。它不是独立的 upstream `omni-mcp` 示例工程；SAGE 生成、预览图和 physics critic 都通过这里的 socket 服务与 Isaac Sim 通信。

## 作用

- `isaac.sim.mcp_extension/`：Isaac Sim 扩展，启动后在本机监听 MCP socket。
- `isaac_mcp/server.py`：Python 侧 MCP 客户端封装，用于向 Isaac Sim 发送命令。
- 默认非 SLURM 端口通常为 `11323`；SLURM 环境会根据 `SLURM_JOB_ID` 派生端口。启动日志永远是端口来源的准确信息。

## 启动

SAGE 当前按 Isaac Sim 5.1 验证。源码版 Isaac Sim 请使用仓库里的 wrapper，不要直接调用 Isaac Sim 自带的 `isaac-sim.sh`。

```bash
cd /home/gaok/coding/sage
conda activate sage5080
./client/isaac_sim_conda.sh \
  --no-window \
  --experience isaacsim.exp.base.kit \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```

成功启动后，日志应包含类似内容：

```text
Isaac Sim MCP server started on localhost:11323
```

`--experience isaacsim.exp.base.kit` 需要保留。直接启动 `isaacsim.exp.full.kit` 通常只会打开普通 Isaac Sim，不会加载 SAGE 的 MCP socket；`client/isaac_sim_conda.sh` 会直接调用 `kit/kit`，确保 base experience 生效。

## 端口

如需固定端口，可在 Isaac Sim 服务端和 Python 客户端两侧使用相同环境变量：

```bash
ISAAC_MCP_PORT=11324 ./client/isaac_sim_conda.sh \
  --no-window \
  --experience isaacsim.exp.base.kit \
  --ext-folder /home/gaok/coding/sage/server/isaacsim \
  --enable isaac.sim.mcp_extension
```

也支持 `SAGE_ISAAC_MCP_PORT`。检查端口：

```bash
ss -ltnp | grep 11323
```

## 关闭服务

前台启动时按 `Ctrl+C` 即可。后台服务可按端口查找并关闭：

```bash
ss -ltnp | grep 11323
kill <PID>
```

如果使用了自定义端口，把 `11323` 替换成实际端口。

## 碰撞近似

常规生成默认使用 `convexDecomposition`：

```bash
SAGE_COLLISION_APPROXIMATION=convexDecomposition
```

可临时改成 `convexHull`：

```bash
SAGE_COLLISION_APPROXIMATION=convexHull ./client/isaac_sim_conda.sh ...
```

不要在常规生成里使用 `sdf`。如果日志出现 `cudaErrorIllegalAddress`、`createSDFBuilder` 或 PhysX GPU SDF cooking 相关崩溃，优先确认没有覆盖成 `sdf`，并确认启动日志路径是 `Isaac-Sim Base/5.1`。

## 常见问题

### Address already in use

说明同一端口已经有 Isaac MCP 服务在运行。先查端口并关闭旧进程，或用 `ISAAC_MCP_PORT` / `SAGE_ISAAC_MCP_PORT` 换一个端口启动。

```bash
ss -ltnp | grep 11323
```

### 没有生成 USD 或预览图失败

先确认 `get_scene_info` 能连通 MCP 服务，再检查 Isaac 日志里对应 handler 的错误。当前单房间生成接口会返回 `usd_file_path`；如果响应里没有该字段，通常说明 handler 没有执行成功或客户端连到了旧服务进程。

### Boolean operation failed

当前墙体开门/开窗不再依赖 trimesh boolean difference，而是拆分墙段生成开口。如果仍看到这个错误，优先确认运行的是当前仓库代码，并重启 Isaac Sim MCP 服务，避免旧扩展进程继续占用端口。
