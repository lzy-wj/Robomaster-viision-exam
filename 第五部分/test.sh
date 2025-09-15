#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[测试] 一键运行 tracking_sim，持续 60 秒（同时保存日志到 test_output.log）"
if [[ -x ./rm_ws/run.sh ]]; then
  bash ./rm_ws/run.sh --duration 60 2>&1 | tee test_output.log || true
else
  echo "[错误] 缺少 rm_ws/run.sh，无法执行测试。" >&2
  exit 1
fi

echo "[提示] 日志已保存到 ./test_output.log；请观察坐标输出与可视化窗口。"
echo "[OK] 第五部分测试完成。"

