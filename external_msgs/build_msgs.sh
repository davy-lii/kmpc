#!/usr/bin/env bash
# 在 kmpc 容器内编译 external_msgs 下所有接口包，装进
# /home/nvidia/workspaces/kmpc/install。然后重新 source setup_real.sh。
set -euo pipefail

_THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_KMPC_WS="$(cd "${_THIS}/.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[external_msgs] 在 kmpc 容器里跑这个脚本，不要在主机上编。" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
unset COLCON_CURRENT_PREFIX

cd "${_KMPC_WS}"
# 只扫 external_msgs，不会碰到 .deps/xoip_overlay，也不会重编 kmpc_cpp。
colcon build --symlink-install --base-paths "${_THIS}"

echo "[external_msgs] 编好了。新开终端或再 source："
echo "  source ${_KMPC_WS}/kmpc_cpp/real/setup_real.sh"
