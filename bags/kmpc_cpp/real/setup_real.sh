#!/usr/bin/env bash
# Thor / 真机 overlay。不要 source 笔记本上的 huacheng_sdk x86 包。
# 每个新终端：
#   source /home/nvidia/workspaces/kmpc/bags/kmpc_cpp/real/setup_real.sh
#
# 本机开发机若 overlay 在仓库 install/kmpc_cpp，把 KMPC_WS 指到仓库根即可。

_KMPC_THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_KMPC_WS="$(cd "${_KMPC_THIS}/../.." && pwd)"
# 源码在 bags/kmpc_cpp/real 时，上两级是 bags，再上一层才是仓库根。
if [[ ! -f "${_KMPC_WS}/install/setup.bash" ]]; then
  _KMPC_WS="$(cd "${_KMPC_WS}/.." && pwd)"
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash || return 1 2>/dev/null || exit 1

# 外部接口消息源码在 ${_KMPC_WS}/external_msgs，编进本工作区 install。
# 加新 .msg：拷到 external_msgs/<pkg>，容器内跑 external_msgs/build_msgs.sh。
if [[ -f "${_KMPC_WS}/install/setup.bash" ]]; then
  unset COLCON_CURRENT_PREFIX
  # shellcheck disable=SC1091
  source "${_KMPC_WS}/install/setup.bash"
fi

# 消息 overlay 必须最后加载；工作区 setup.bash 的旧 prefix chain 会覆盖消息库。
# 消息 overlay（aarch64）。kmpc/install 的 prefix chain 只有 humble，
# 新开的 kmpc 容器不会自带 PYTHONPATH，必须显式 source。不要 source 华成驱动包。
# 20261103 新版优先用 SDK 20260930 overlay；没有再退回旧 xoip_overlay。
_KMPC_XOIP=""
if [[ -d "${_KMPC_WS}/.deps/xoip_overlay_20260930/xoip_ros2" ]]; then
  _KMPC_REVIEW_SDK="${_KMPC_WS}/.deps/xoip_overlay_20260930/xoip_ros2"
  export AMENT_PREFIX_PATH="${_KMPC_REVIEW_SDK}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
  export LD_LIBRARY_PATH="${_KMPC_REVIEW_SDK}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTHONPATH="${_KMPC_REVIEW_SDK}/local/lib/python3.10/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
  unset _KMPC_REVIEW_SDK
  echo "[setup_real] xoip overlay -> ${_KMPC_WS}/.deps/xoip_overlay_20260930"
elif [[ -f "${_KMPC_WS}/.deps/xoip_overlay/setup.bash" ]]; then
  _KMPC_XOIP="${_KMPC_WS}/.deps/xoip_overlay/setup.bash"
  unset COLCON_CURRENT_PREFIX
  # shellcheck disable=SC1091
  source "${_KMPC_XOIP}"
  echo "[setup_real] xoip overlay -> ${_KMPC_XOIP}"
fi
unset _KMPC_XOIP

# 真机节点装在仓库根的 kmpc_cpp/，没有挂进 install/ 的 prefix 链。
_KMPC_PREFIX="${_KMPC_WS}/kmpc_cpp"
if [[ -f "${_KMPC_PREFIX}/share/ament_index/resource_index/packages/kmpc_cpp" ]]; then
  export AMENT_PREFIX_PATH="${_KMPC_PREFIX}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
  export CMAKE_PREFIX_PATH="${_KMPC_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"   # 板上 Docker 之间约定 20
unset ROS_LOCALHOST_ONLY
# 与 jetpack / integrate / hand 相同：只走有线口 192.168.4.2。
# start_kmpc.sh 用 --rcfile 启动，不会读容器 ~/.bashrc，所以必须在这里导出。
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${_KMPC_THIS}/fastdds_wired.xml}"

if ! python3 - <<'PY'
from rclpy.type_support import check_for_type_support
from xoip_ros2.msg import MultiJointServoCommand, StreamStatus
check_for_type_support(MultiJointServoCommand)
check_for_type_support(StreamStatus)
print("[setup_real] OK — MultiJointServoCommand StreamStatus")
PY
then
  echo "[setup_real] 找不到 aarch64 xoip_ros2（MultiJointServoCommand）。在容器里 source 镜像自带 overlay，不要拷笔记本 x86 的 install/xoip_ros2。" >&2
  return 1 2>/dev/null || exit 1
fi

if ! ros2 pkg prefix kmpc_cpp >/dev/null 2>&1; then
  echo "[setup_real] 还没有 kmpc_cpp。先 colcon build（见 Thor部署/真机Thor部署注意事项.md）" >&2
  return 1 2>/dev/null || exit 1
fi

echo "[setup_real] OK — kmpc_cpp at $(ros2 pkg prefix kmpc_cpp)  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-20}"
echo "[setup_real] FastDDS ${FASTRTPS_DEFAULT_PROFILES_FILE}"
if ros2 pkg prefix eye_msgs >/dev/null 2>&1; then
  echo "[setup_real] OK — eye_msgs at $(ros2 pkg prefix eye_msgs)"
else
  echo "[setup_real] 还没有 eye_msgs（/planning/eye_state）。容器内: bash ${_KMPC_WS}/external_msgs/build_msgs.sh" >&2
fi
echo "[setup_real] 先看姿态：ros2 launch kmpc_cpp kmpc_cpp_viz.launch.py"
echo "[setup_real] 再控制：  ros2 launch kmpc_cpp kmpc_cpp_real.launch.py"
