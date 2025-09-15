#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[3.1] 并发通信"
bash 3.1_async_comm/test.sh

echo "[3.2] 插件化技能管理器（基础）"
bash 3.2_skills/test.sh

echo "[3.2] 插件化技能管理器（高级）"
bash 3.2_skills/test_advanced.sh

echo "[3.3] 日志流式统计"
bash 3.3_logstream/test.sh

echo "[OK] 第三部分全部测试通过。"


