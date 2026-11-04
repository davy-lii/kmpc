# 位置刷新周期接口使用与排错

本文整理 `/xoip/robot/get_pos_refresh_ms` 和 `/xoip/robot/set_pos_refresh_ms` 的使用方法，以及本次在 `root@nvidia:/ros2_ws` 环境中遇到的接口加载问题。

## 1. 接口用途

这两个接口都是 ROS 2 Service，通过请求中的 `robot_id` 选择机器人。

| 服务名称 | 服务类型 | 用途 |
| --- | --- | --- |
| `/xoip/robot/get_pos_refresh_ms` | `xoip_ros2/srv/GetPosRefresh` | 查询位置刷新周期 |
| `/xoip/robot/set_pos_refresh_ms` | `xoip_ros2/srv/SetPosRefresh` | 设置位置刷新周期 |

它们分别映射到 SDK 的 `getPosReFreshMS` 和 `setPosReFreshMS`。

### 查询接口

```text
int32 robot_id
---
bool success
string message
uint16 joint_ms
uint16 world_ms
```

### 设置接口

```text
int32 robot_id
uint16 joint_ms
uint16 world_ms
---
bool success
string message
```

| 字段 | 含义 |
| --- | --- |
| `robot_id` | 目标机器人 ID，例如 `1` |
| `joint_ms` | 关节位置刷新周期，单位 ms |
| `world_ms` | 世界坐标位姿刷新周期，单位 ms |
| `success` | 本次操作是否成功 |
| `message` | 操作结果说明，失败时用于排查 |

周期与频率的换算为 `频率 Hz = 1000 / 周期 ms`。例如，20 ms 对应 50 Hz，10 ms 对应 100 Hz。

## 2. 使用步骤

前提：当前终端已加载包含上述接口的包环境，服务端节点已启动并连接控制器。

先查询机器人 1 的当前设置：

```bash
ros2 service call /xoip/robot/get_pos_refresh_ms \
  xoip_ros2/srv/GetPosRefresh \
  "{robot_id: 1}"
```

确认响应的 `success` 为 `true`，再读取 `joint_ms` 和 `world_ms`。

如需将两个周期都设为 20 ms：

```bash
ros2 service call /xoip/robot/set_pos_refresh_ms \
  xoip_ros2/srv/SetPosRefresh \
  "{robot_id: 1, joint_ms: 20, world_ms: 20}"
```

设置后再次调用查询接口，确认实际值。如果仅修改一个周期，应先查询另一个周期，并把原值一起传入。不要省略周期字段：省略的数值字段默认为 0，而当前项目文档未说明 0 的语义。

### 与 ROS topic 发布频率的关系

位置刷新周期和 ROS 状态 topic 发布频率是不同配置。节点通过 `state_publish_rate` 参数控制状态发布频率，当前项目示例配置为 `50.0` Hz。把位置刷新周期设置为 10 ms，不代表 topic 会自动变为 100 Hz。

当前资料未明确说明控制器支持的最小周期、0 的含义或配置是否在重启后保留。`uint16` 的数值表示范围不能直接当作控制器支持范围。

## 3. 本次故障与已确认原因

最初调用报错：

```text
The passed service type is invalid
```

加载工作空间后检查：

```bash
source /ros2_ws/install/setup.bash
ros2 pkg prefix xoip_ros2
ros2 interface show xoip_ros2/srv/GetPosRefresh
```

输出表明当前选中的是：

```text
/ros2_ws/install/xoip_ros2
```

但该目录找不到 `GetPosRefresh.idl`，同时：

```bash
python3 -c "from xoip_ros2.srv import GetPosRefresh; print(GetPosRefresh)"
```

报 `ImportError`，导入位置也是 `/ros2_ws/install/xoip_ros2/local/lib/python3.10/dist-packages/xoip_ros2/srv/__init__.py`。

用户提供的文件搜索结果显示，以下目录包含这两个服务的接口文件：

```text
/ros2_ws/src/xoip_ros2/share/xoip_ros2/srv/
/ros2_ws/src/huacheng_sdk/20260908/install/xoip_ros2/share/xoip_ros2/srv/
/ros2_ws/src/huacheng_sdk/20260824/install/xoip_ros2/share/xoip_ros2/srv/
```

因此，已确认的问题是：当前终端选中的安装包没有这两个接口，而其他目录存在包含接口文件的包。服务名称和请求写法没有发现错误。

这些搜索命中属于安装产物结构，不能仅因为它们位于 `src` 下，就当作可以直接执行 `colcon build` 的源码包。也不能只复制 `.idl` 文件解决调用问题：Python 接口和对应类型支持库也必须可用。

## 4. 环境切换与验证

在同一个调用终端中，先加载工作空间，再加载目标包环境：

```bash
source /ros2_ws/install/setup.bash
source /ros2_ws/src/xoip_ros2/share/xoip_ros2/local_setup.bash
```

验证选中的包和接口：

```bash
ros2 pkg prefix xoip_ros2
python3 -c "from xoip_ros2.srv import GetPosRefresh; print(GetPosRefresh)"
ros2 interface show xoip_ros2/srv/GetPosRefresh
ros2 interface show xoip_ros2/srv/SetPosRefresh
```

预期包路径为 `/ros2_ws/src/xoip_ros2`，Python 导入成功，并能显示两个服务的完整定义。随后重试第 2 节的查询命令。

加载顺序很重要：目标包环境应最后加载。如果随后再次加载 `/ros2_ws/install/setup.bash`，可能重新选中缺少接口的包。新终端也需要加载环境。

截至本文整理时，已获得接口缺失和多份包并存的诊断输出；尚未收到环境切换后的验证结果，也未确认真实控制器上的查询或设置成功。

## 5. 后续排查

### 停在等待服务

如果输出变为：

```text
waiting for service to become available...
```

检查当前 ROS 图中是否发现服务：

```bash
ros2 service list -t | grep pos_refresh
```

若没有输出，检查服务端节点是否运行、其版本是否提供这两个接口，以及客户端与服务端的 ROS 发现环境是否一致（例如 `ROS_DOMAIN_ID`）。切换客户端终端的环境不会更新已经运行的服务端进程；若服务端使用缺少接口的版本，需要在正确环境中重新启动相应节点。

### Python 导入仍失败

保留完整报错，并检查：

```bash
ros2 pkg prefix xoip_ros2
python3 --version
uname -m
python3 -c "import xoip_ros2; print(xoip_ros2.__file__)"
```

本次本地检查的分发包包含 Python 3.10 / aarch64 二进制扩展，运行环境需匹配。文件中存在 `.srv` 或 `.idl`，并不单独证明二进制类型支持能成功加载。

## 6. 项目内参考

- `share/xoip_ros2/srv/GetPosRefresh.srv`
- `share/xoip_ros2/srv/SetPosRefresh.srv`
- `share/xoip_ros2/docs/USER_GUIDE.md`：节点参数和服务说明
- `share/xoip_ros2/docs/API_COVERAGE.md`：SDK 接口映射
