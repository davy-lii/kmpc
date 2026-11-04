# kmpc 外部接口消息

kmpc 容器不挂 `robot_brain`。需要 echo / 订阅板外 topic 时，把对应 ROS 接口包放在这里，在容器里编译进 `workspaces/kmpc/install`。

`./start_kmpc.sh` 仍然只 source `kmpc_cpp/real/setup_real.sh`，那个脚本会 source `install/setup.bash`，编过的包会自动进环境。不要再拷 `install` overlay。

## 现在有的包

| 包 | 用途 |
|---|---|
| `eye_msgs` | `/planning/eye_state`、`/planning/eye_cmd` |

华成驱动消息仍用 `.deps/xoip_overlay`（aarch64 预编译），不要搬进这里。

## 加新消息

1. 把接口包拷到本目录，和 `eye_msgs` 平级，包内要有 `package.xml`、`CMakeLists.txt`、`msg/`。
2. 进 kmpc 容器：

```bash
bash /home/nvidia/workspaces/kmpc/external_msgs/build_msgs.sh
source /home/nvidia/workspaces/kmpc/kmpc_cpp/real/setup_real.sh
```

3. 改了 `.msg` 后再跑一遍 `build_msgs.sh`，已经打开的终端要重新 source。
