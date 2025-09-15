#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[2.1] 开发环境自动化"
bash 2.1_env/test.sh

echo "[2.2] 工程版本控制"
bash 2.2_repo/test.sh

echo "[2.3] 应用容器化"
bash 2.3_container/test.sh

echo "[OK] 第二部分全部测试通过。"


