# -*- coding: utf-8 -*-
"""
海量日志的流式统计：
- 使用生成器逐行读取，常数级内存统计包含关键词 "ERROR" 的行数。

数学思考：在概率模型中，可将每行是否包含 ERROR 视为伯努利变量 X_i，
总计数 S_n = sum X_i。生成器保证内存 O(1)，时间复杂度 O(n)。
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterable, Iterator


def open_text(path: Path) -> Iterable[str]:
    """透明打开 .gz 或普通文本文件，逐行生成。"""
    if str(path).endswith(".gz"):
        with gzip.open(path, mode="rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
    else:
        with path.open("rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line


def count_error_lines(lines: Iterator[str], keyword: str = "ERROR") -> int:
    count = 0
    for line in lines:
        if keyword in line:
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="流式统计包含 ERROR 的日志行数")
    parser.add_argument("logfile", nargs="?", type=str, help="日志文件路径（可为 .gz）")
    parser.add_argument("--keyword", type=str, default="ERROR", help="匹配关键字")
    args = parser.parse_args()

    # 无参数时，按约定尝试当前目录下的默认文件名
    if args.logfile is None:
        candidates = ["robot.log", "robot.log.gz", "sample.log"]
        for name in candidates:
            p = Path(name)
            if p.exists():
                path = p
                break
        else:
            print("未提供日志文件且未找到默认文件（robot.log / robot.log.gz / sample.log）。")
            return 2
    else:
        path = Path(args.logfile)
        if not path.exists():
            print(f"文件不存在：{path}")
            return 2

    lines = open_text(path)
    total = count_error_lines(iter(lines), keyword=args.keyword)
    print(total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


