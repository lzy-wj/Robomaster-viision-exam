#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[测试] setup_env.sh --check"
bash setup_env.sh --check
echo "[OK] 环境检测脚本可运行。"


