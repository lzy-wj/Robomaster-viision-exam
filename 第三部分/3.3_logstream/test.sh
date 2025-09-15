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

  # 继续尝试在当前目录查找 robot.zip 并自动解压校验
  if [ -f "robot.zip" ]; then
    echo "[信息] 发现 robot.zip，尝试解压..."
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
      echo "[信息] 使用解压得到的日志：$CANDIDATE"
      CNT2=$($PY stream_count_error.py "$CANDIDATE")
      if [[ "$CANDIDATE" == *.gz ]]; then
        EXPECT2=$(gzip -cd "$CANDIDATE" | grep -c "ERROR" || true)
      else
        EXPECT2=$(grep -c "ERROR" "$CANDIDATE" || true)
      fi
      echo "[信息] 期望计数(grep)=$EXPECT2, 脚本输出=$CNT2"
      test "$CNT2" = "$EXPECT2"
      echo "[OK] robot.zip 日志校验通过。"
    else
      echo "[警告] 未在 robot.zip 中找到可用的日志文件（robot.log/*.log/*.log.gz）。"
    fi
  else
    echo "[提示] 未发现 robot.zip，跳过大日志校验。"
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

echo "下载与测试大文件的提示："
echo "- 请下载压缩包：https://github.com/cygnomatic/Cygnomatic-Site/raw/refs/heads/main/docs/robot.zip?download="
echo "- 将 robot.zip 放在当前目录后，直接运行：bash test.sh  即可自动解压并校验"


