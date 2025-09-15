#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "[警告] 未检测到 docker，跳过容器构建与运行测试。"
  exit 0
fi

IMAGE_TAG="rm-hello:latest"

echo "[测试] docker build"
docker build -t "$IMAGE_TAG" .

echo "[测试] docker run"
docker run --rm "$IMAGE_TAG" | grep -q "Hello, RoboMaster!"

echo "[OK] task3_container 测试通过。"


