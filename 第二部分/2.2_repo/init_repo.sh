#!/usr/bin/env bash
set -euo pipefail

# 在指定目录创建一个基础 Git 项目（Hello, RoboMaster!），并配置 .gitignore。

DEST_DIR="${1:-sample_repo}"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# 目录结构
mkdir -p hello_rm/src hello_rm/tests

cat > hello_rm/src/main.py <<'PY'
# -*- coding: utf-8 -*-
"""最小可运行程序：打印 Hello, RoboMaster!"""

def main() -> None:
    print("Hello, RoboMaster!")


if __name__ == "__main__":
    main()
PY

cat > hello_rm/README.md <<'MD'
### Hello, RoboMaster!

运行方式：
```bash
python hello_rm/src/main.py
```
MD

cat > .gitignore <<'GI'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# Envs
.venv/
venv/
env/

# OS
.DS_Store
Thumbs.db

# Editors
.idea/
.vscode/
GI

if [ ! -d .git ]; then
  git init -q
fi

git add -A
if ! git diff --cached --quiet; then
  git commit -m "init: Hello, RoboMaster! 基础项目结构" >/dev/null 2>&1 || true
fi

echo "[OK] 基础 Git 项目已生成于：$(pwd)"


