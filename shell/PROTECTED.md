# 禁止发布覆盖

本目录（`workspaces/kmpc/shell/`）是 **kmpc 容器交互 ROS 环境**，给 `./start_kmpc.sh` / `ros2 topic` 用。

发布 `deploy/activate` 只允许动：

- `releases/<ver>/kmpc/`
- `releases/current`
- `deployment.entries` 里那几条 symlink（`kmpc_cpp/lib|share`、`install/whole_body_kmpc/share`）
- `ljw` 下被 checksum 管理的启动脚本

**不要**把 `shell/`、`kmpc_cpp/real/setup_real.sh`（兼容软链）、本文件写进 entries，也不要 rmtree 本目录。
