#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 若传入外部日志路径，则使用之；否则生成一个最小示例文件
if [ "${1-}" != "" ]; then
  LOG_PATH="$1"
  echo "[信息] 使用外部日志路径：$LOG_PATH"
else
  LOG_PATH="robot.log"
  rm -f "$LOG_PATH"
  {
    echo "INFO start";
    echo "WARN something";
    echo "ERROR first";
    echo "INFO mid";
    echo "ERROR second";
  } > "$LOG_PATH"
fi

if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

CNT=$($PY stream_count_error.py "$LOG_PATH")

# 如果是内置示例，则断言计数=2；
if [ "$LOG_PATH" = "robot.log" ]; then
  test "$CNT" = "2"
  echo "[OK] task6_logstream 基础测试通过（示例文件）。"

  # 大文件测试逻辑：优先使用原始数据，否则生成测试数据
  echo "[信息] 开始大文件日志测试..."
  
  if [ -f "robot.zip" ]; then
    echo "[信息] 发现原始 robot.zip 文件，使用原始数据进行测试"
    EXTRACT_DIR="extracted_robot"
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    if command -v unzip >/dev/null 2>&1; then
      unzip -o -q robot.zip -d "$EXTRACT_DIR"
    else
      echo "[提示] 未找到 unzip，改用 Python 解压"
      $PY - <<'PY'
import sys, zipfile
zf = zipfile.ZipFile('robot.zip')
zf.extractall('extracted_robot')
print('[信息] 已使用 Python 解压到 extracted_robot')
PY
    fi

    # 搜索候选日志（优先 robot.log / *.log.gz / *.log），排除当前示例 robot.log
    CANDIDATE=$(find "$EXTRACT_DIR" -type f \( -name 'robot.log' -o -name '*.log.gz' -o -name '*.log' \) | grep -v "^./robot.log$" | head -n 1 || true)
    if [ "${CANDIDATE-}" != "" ]; then
      echo "[信息] 使用解压得到的原始日志：$CANDIDATE"
      CNT2=$($PY stream_count_error.py "$CANDIDATE")
      if [[ "$CANDIDATE" == *.gz ]]; then
        EXPECT2=$(gzip -cd "$CANDIDATE" | grep -c "ERROR" || true)
      else
        EXPECT2=$(grep -c "ERROR" "$CANDIDATE" || true)
      fi
      echo "[信息] 期望计数(grep)=$EXPECT2, 脚本输出=$CNT2"
      test "$CNT2" = "$EXPECT2"
      echo "[OK] 原始 robot.zip 日志校验通过！"
    else
      echo "[警告] 未在 robot.zip 中找到可用的日志文件，尝试生成测试数据..."
      NEED_GENERATE=true
    fi
  else
    echo "[信息] 未发现原始 robot.zip 文件，生成测试数据进行验证"
    NEED_GENERATE=true
  fi
  
  # 如果需要生成测试数据
  if [ "${NEED_GENERATE-}" = "true" ]; then
    if [ -f "generate_test_data.py" ]; then
      echo "[信息] 调用数据生成脚本创建测试日志..."
      $PY generate_test_data.py
      
      # 查找生成的测试文件
      if [ -f "robot_test.log.gz" ]; then
        TEST_LOG="robot_test.log.gz"
        echo "[信息] 使用生成的测试日志：$TEST_LOG"
        CNT3=$($PY stream_count_error.py "$TEST_LOG")
        EXPECT3=$(gzip -cd "$TEST_LOG" | grep -c "ERROR" || true)
        echo "[信息] 期望计数(grep)=$EXPECT3, 脚本输出=$CNT3"
        test "$CNT3" = "$EXPECT3"
        echo "[OK] 生成的测试日志校验通过！"
      else
        echo "[警告] 测试数据生成失败，跳过大日志测试"
      fi
    else
      echo "[警告] 未找到 generate_test_data.py 脚本，无法生成测试数据"
    fi
  fi
else
  # 如果是外部文件，则用 grep 计算期望值并进行对比校验
  if [[ "$LOG_PATH" == *.gz ]]; then
    EXPECT=$(gzip -cd "$LOG_PATH" | grep -c "ERROR" || true)
  else
    EXPECT=$(grep -c "ERROR" "$LOG_PATH" || true)
  fi
  echo "[信息] 期望计数(grep)=$EXPECT, 脚本输出=$CNT"
  test "$CNT" = "$EXPECT"
  echo "[OK] 外部日志校验通过。"
fi

echo "💡 大文件测试使用说明："
echo "方式1（推荐）：如果您有原始的1GB日志文件"
echo "  - 下载：https://github.com/cygnomatic/Cygnomatic-Site/raw/refs/heads/main/docs/robot.zip?download="
echo "  - 将 robot.zip 放在当前目录，运行 bash test.sh 即可自动测试"
echo ""
echo "方式2（自动备选）：如果没有原始文件"
echo "  - 脚本会自动调用 generate_test_data.py 生成30MB测试日志"
echo "  - 无需手动操作，测试脚本会智能选择合适的数据源"


