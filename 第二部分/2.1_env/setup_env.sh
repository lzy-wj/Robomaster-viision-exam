#!/usr/bin/env bash
set -euo pipefail

# 中文注释与输出，数学化/工程化思维：
# 目标：自动化安装/校验机器人开发所需工具（git, cmake, docker 等），
#      处理权限（docker 组）、架构识别与幂等性。

print_usage() {
  cat <<'USAGE'
用法：
  bash setup_env.sh --check        # 仅检测并打印工具与权限状态
  bash setup_env.sh --install      # 使用 apt 系安装并配置权限（需要 sudo）

说明：
  - 当前脚本优先支持基于 Debian/Ubuntu 的发行版（apt）。
  - 会检测 CPU 架构（uname -m），提示潜在兼容性。
  - 对 docker：安装后将把当前用户加入 docker 组（需重新登录生效）。
USAGE
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

detect_arch() {
  uname -m || echo "unknown"
}

detect_distro() {
  if need_cmd apt-get; then
    echo "debian"
  elif need_cmd dnf; then
    echo "fedora"
  elif need_cmd pacman; then
    echo "arch"
  else
    echo "unknown"
  fi
}

print_versions() {
  echo "[信息] CPU 架构：$(detect_arch)" || true
  if need_cmd lsb_release; then lsb_release -a || true; fi
  if need_cmd git; then git --version || true; else echo "git: 未安装"; fi
  if need_cmd cmake; then cmake --version | head -n1 || true; else echo "cmake: 未安装"; fi
  if need_cmd docker; then docker --version || echo "docker: 已安装但无法执行"; else echo "docker: 未安装"; fi
  if need_cmd docker; then
    if groups "$USER" 2>/dev/null | grep -q '\bdocker\b'; then
      echo "[信息] 用户已在 docker 组内：$USER"
    else
      echo "[警告] 用户不在 docker 组内，安装后将配置（需重新登录）"
    fi
  fi
}

ensure_sudo() {
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if ! need_cmd sudo; then
      echo "[错误] 需要 root 或 sudo 执行安装。" >&2
      exit 1
    fi
  fi
}

install_debian() {
  ensure_sudo
  # 基础工具与证书
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates curl gnupg lsb-release \
    git cmake build-essential

  # docker（优先使用官方仓库；若失败则回退到 docker.io）
  if ! need_cmd docker; then
    if need_cmd curl && need_cmd gpg; then
      sudo install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg || true
      sudo chmod a+r /etc/apt/keyrings/docker.gpg || true
      codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
      echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $codename stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null || true
      sudo apt-get update -y || true
      sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || \
      sudo apt-get install -y docker.io
    else
      sudo apt-get install -y docker.io
    fi
  fi

  # 将当前用户加入 docker 组（若存在）
  if getent group docker >/dev/null 2>&1; then
    sudo usermod -aG docker "$USER" || true
    echo "[信息] 已尝试将 $USER 加入 docker 组，需重新登录生效。"
  fi
}

main() {
  if [ "$#" -lt 1 ]; then
    print_usage; exit 1
  fi

  case "$1" in
    --check)
      print_versions
      ;;
    --install)
      case "$(detect_distro)" in
        debian)
          install_debian
          ;;
        fedora|arch)
          echo "[错误] 暂未实现该发行版的自动安装，请手动安装 git cmake docker。" >&2
          exit 2
          ;;
        *)
          echo "[错误] 未识别的系统/不受支持的包管理器，请在 Debian/Ubuntu 上运行或手动安装。" >&2
          exit 2
          ;;
      esac
      ;;
    -h|--help)
      print_usage
      ;;
    *)
      print_usage; exit 1
      ;;
  esac
}

main "$@"


