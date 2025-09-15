#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEST="sample_repo"
rm -rf "$DEST"

echo "[测试] 生成基础 Git 项目"
bash init_repo.sh "$DEST"

echo "[测试] 校验文件存在"
test -f "$DEST/hello_rm/src/main.py"
test -f "$DEST/.gitignore"

echo "[测试] 校验 Git 仓库"
cd "$DEST"
git rev-parse --is-inside-work-tree >/dev/null 2>&1

echo "[测试] 运行程序"
if command -v python >/dev/null 2>&1; then
  python hello_rm/src/main.py | grep -q "Hello, RoboMaster!"
elif command -v python3 >/dev/null 2>&1; then
  python3 hello_rm/src/main.py | grep -q "Hello, RoboMaster!"
else
  echo "[警告] 未检测到 Python，跳过运行校验。"
fi

echo "[OK] task2_repo 测试通过。"


