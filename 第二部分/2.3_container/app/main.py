# -*- coding: utf-8 -*-
"""容器化最小应用：打印 Hello, RoboMaster!（可选参数）。"""

import sys


def main() -> None:
    name = "RoboMaster"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    print(f"Hello, {name}!")


if __name__ == "__main__":
    main()


