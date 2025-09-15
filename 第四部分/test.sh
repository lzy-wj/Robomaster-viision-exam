#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[信息] 第四部分测试脚本 (可选CLI模式)"

# CLI直通模式：
# 用法1：bash test.sh <start_x> <start_y> <goal_x> <goal_y>    # 使用默认 astar/map.txt
# 用法2：bash test.sh <map_path> <start_x> <start_y> <goal_x> <goal_y>
if [ "$#" -eq 4 ] || [ "$#" -eq 5 ]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  else
    PY=python3
  fi
  if [ "$#" -eq 4 ]; then
    MAP="astar/map.txt"; SX=$1; SY=$2; GX=$3; GY=$4
  else
    MAP=$1; SX=$2; SY=$3; GX=$4; GY=$5
  fi
  echo "[运行] A* CLI: $MAP $SX $SY $GX $GY"
  # 记录到两个常见文件，便于查阅
  OUT_ROOT="astar_out.txt"
  $PY astar/astar.py "$MAP" "$SX" "$SY" "$GX" "$GY" | tee "$OUT_ROOT"
  exit 0
fi

echo "[测试] 第四部分：基础算法"
pushd algorithms >/dev/null
if command -v python >/dev/null 2>&1; then
  python tests.py
else
  python3 tests.py
fi
popd >/dev/null

echo "[测试] A* 路径规划 CLI（示例：不可达路径，应输出提示）"
if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

# 运行题目指定示例：map.txt 从 (0,0) 到 (4,4)
OUT_FILE="astar_out.txt"
$PY astar/astar.py astar/map.txt 0 0 4 4 | tee "$OUT_FILE"

# 由于该示例不可达，检查提示语是否出现
grep -q "postion (4,4)" "$OUT_FILE"

# 增加一个可达用例：从 (0,0) 到 (0,2) 应该存在直线路径并打印带 * 的地图
echo "[测试] A* 路径规划 CLI（示例：可达路径，应打印 * 路径）"
OUT_FILE_OK="astar_out_ok.txt"
$PY astar/astar.py astar/map.txt 0 0 0 2 | tee "$OUT_FILE_OK"
# 断言输出中包含 *，且不包含不可达提示
grep -q "\*" "$OUT_FILE_OK"
! grep -q "postion (0,2)" "$OUT_FILE_OK"

echo "[OK] 第四部分测试通过。"


