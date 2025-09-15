#!/usr/bin/env bash
set -euo pipefail

# 一键运行：自动 source ROS2、编译并启动 tracking_sim.launch.py
# 可选：--duration SEC 指定运行秒数（使用 timeout），未指定则持续运行直到 Ctrl-C。

cd "$(dirname "$0")"

DURATION=""
if [[ ${1:-} == "--duration" && -n ${2:-} ]]; then
  DURATION="$2"
  shift 2
fi

echo "[信息] 尝试加载 ROS2 环境"
# 某些 ROS setup 脚本依赖未设置的变量，需临时关闭 nounset
set +u
if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # 已有 ROS_DISTRO，优先使用
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
elif [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source /opt/ros/jazzy/setup.bash
elif [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source /opt/ros/humble/setup.bash
else
  echo "[错误] 未找到 ROS2 安装，请先在本机安装并确保 /opt/ros/<distro>/setup.bash 存在。" >&2
  exit 1
fi
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[错误] 未检测到 ros2 命令，请确认已正确 source 对应 setup.bash。" >&2
  exit 1
fi

echo "[步骤] colcon build"
# 优先使用 --log-base（若此 colcon 版本支持）；否则构建后清理 rm_ws/log
if colcon build --help 2>/dev/null | grep -q -- '--log-base'; then
  TMP_LOG_BASE=$(mktemp -d -t colcon_logs.XXXXXX 2>/dev/null || mktemp -d)
  colcon build --event-handlers console_cohesion+ --parallel-workers 1 --log-base "$TMP_LOG_BASE" | cat
  rm -rf "$TMP_LOG_BASE" || true
else
  colcon build --event-handlers console_cohesion+ --parallel-workers 1 | cat
  rm -rf log || true
fi

echo "[步骤] source 本工作区安装环境"
# shellcheck disable=SC1091
set +u
source install/setup.bash
set -u

# 兼容修复：某些环境下 ament 没有创建 Python 包的 libexec 目录，导致 launch 报错
BIN_DIR="install/rm_vision_sim/bin"
LIBEXEC_DIR="install/rm_vision_sim/lib/rm_vision_sim"
if [[ -d "$BIN_DIR" && ! -d "$LIBEXEC_DIR" ]]; then
  echo "[修复] 创建缺失的 libexec 目录并复制可执行脚本"
  mkdir -p "$LIBEXEC_DIR"
  cp -f "$BIN_DIR"/* "$LIBEXEC_DIR"/
fi

# 强制所有 Python 节点无缓冲输出；若存在 stdbuf 则进一步行缓冲
export PYTHONUNBUFFERED=1
export RCUTILS_LOGGING_ENABLE_FILE_LOGGING=0
export RCL_LOGGING_ENABLE_FILE_LOGGING=0
export RCUTILS_LOGGING_BUFFERED_STREAM=0
if command -v stdbuf >/dev/null 2>&1; then
  LAUNCH_CMD=(stdbuf -oL -eL ros2 launch rm_vision_sim tracking_sim.launch.py)
else
  LAUNCH_CMD=(ros2 launch rm_vision_sim tracking_sim.launch.py)
fi

echo "[步骤] 启动 tracking_sim.launch.py"
if [[ -n "$DURATION" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    timeout "${DURATION}s" "${LAUNCH_CMD[@]}" || true
  else
    echo "[警告] 未检测到 timeout，将直接前台运行（Ctrl-C 结束）。"
    "${LAUNCH_CMD[@]}"
  fi
else
  "${LAUNCH_CMD[@]}"
fi

echo "[完成] 运行结束。"


