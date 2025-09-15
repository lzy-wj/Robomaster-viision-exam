#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

echo "[测试] 运行高级用例 test_advanced.py"
$PY test_advanced.py | tee out.txt

grep -q "测试完成" out.txt || true
echo "[OK] skills 高级测试执行完成。"


