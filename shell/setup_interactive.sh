#!/usr/bin/env bash
# kmpc 容器交互壳：查 topic / 调试用。不做 launch、不 enable。
#
# 位置故意放在 workspaces/kmpc/shell/，不在 release 的 deployment.entries 里。
# 发布只允许改 releases/<ver>/ 和 entries 里那几条 symlink；禁止删/覆盖本目录。
#
# 用法：
#   主机: ./start_kmpc.sh
#   容器内: source /home/nvidia/workspaces/kmpc/shell/setup_interactive.sh

_KMPC_SHELL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_KMPC_WS="$(cd "${_KMPC_SHELL}/.." && pwd)"
_KMPC_ENV="${_KMPC_WS}/releases/current/kmpc/env.sh"

if [[ ! -f "${_KMPC_ENV}" ]]; then
  echo "[setup_interactive] 缺少 ${_KMPC_ENV}；先切 current 再进壳" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${_KMPC_ENV}" || return 1 2>/dev/null || exit 1

_KMPC_RELEASE="$(cd "$(dirname "${_KMPC_ENV}")" && pwd -P)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
unset ROS_LOCALHOST_ONLY
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${_KMPC_RELEASE}/fastdds_wired.xml}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[setup_interactive] ros2 仍不在 PATH，env.sh 未生效: ${_KMPC_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

echo "[setup_interactive] release=${_KMPC_RELEASE}"
echo "[setup_interactive] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} FastDDS=${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "[setup_interactive] ros2=$(command -v ros2)  — 例: ros2 topic list | head"
