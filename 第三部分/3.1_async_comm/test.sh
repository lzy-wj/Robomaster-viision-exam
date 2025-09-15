#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 在后台启动三个简单的 nc TCP 服务端
cleanup() {
  set +e
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "nc -l -p 8001" 2>/dev/null || true
    pkill -f "nc -l -p 8002" 2>/dev/null || true
    pkill -f "nc -l -p 8003" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! command -v nc >/dev/null 2>&1; then
  echo "[错误] 未安装 netcat(nc)，无法进行端口服务测试。" >&2
  exit 1
fi

# 兼容不同 nc 版本（BSD vs GNU），优先使用 -l -p 形式
nc -l -p 8001 >/dev/null 2>&1 &
nc -l -p 8002 >/dev/null 2>&1 &
nc -l -p 8003 >/dev/null 2>&1 &
sleep 0.15

echo "[测试] 运行 asyncio 客户端连接"
if command -v python >/dev/null 2>&1; then
  python client.py --targets "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003" --timeout 0.8 | tee out.txt
else
  python3 client.py --targets "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003" --timeout 0.8 | tee out.txt
fi

grep -q "SUMMARY: 3/3 connected" out.txt
echo "[OK] task4_async_comm 测试通过。"


