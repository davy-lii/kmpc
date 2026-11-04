# thor_kmpc — 拷到 Thor 的这一份

本目录是给 **NVIDIA Thor（aarch64）** 的上传包，不是笔记本仿真工作区。

落点：`/home/nvidia/workspaces/kmpc/`

| 文件 | 用途 |
|------|------|
| [Thor部署/](Thor部署/README.md) | **另一台 Thor 从零部署**：环境、Docker、华成 250 Hz yaml、给 AI 的提示词 |
| `kmpc_cpp/` | C++ 控制器源码 + `real/` launch（**无** `sim/`） |
| `whole_body_kmpc/` | yaml / URDF / `pose_gui` |

在**笔记本**上改完代码后重新打包：

```bash
/home/davy/dual_arm_mpc_papers/kmpc_cpp/tools/pack_thor_kmpc.sh
```

然后再把本目录同步到板上。不要在本目录里跑仿真。
