# -*- coding: utf-8 -*-
"""
A* 路径规划（二维栅格，4-邻接，每步代价=1）

数学思考：
- 估价函数 f(n) = g(n) + h(n)，其中 g 为起点到 n 的实际代价，h 为启发式（此处取曼哈顿距离）。
- 当 h 为“一致可采纳”（admissible & consistent），A* 能保证找到最优路径且不对已固定点做无谓扩展。
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Grid = List[List[int]]


def load_map(path: Path) -> Grid:
    grid: Grid = []
    for raw in path.read_text(encoding="utf-8").strip().splitlines():
        row = [int(x) for x in raw.strip().split()]
        grid.append(row)
    if not grid:
        raise ValueError("地图为空")
    # 检查矩形性
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("每行列数不一致")
    return grid


def in_bounds(grid: Grid, x: int, y: int) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def is_free(grid: Grid, x: int, y: int) -> bool:
    return grid[y][x] == 0


def neighbors4(grid: Grid, x: int, y: int) -> Iterable[Tuple[int, int]]:
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if in_bounds(grid, nx, ny) and is_free(grid, nx, ny):
            yield nx, ny


def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def astar(grid: Grid, sx: int, sy: int, gx: int, gy: int) -> Optional[List[Tuple[int, int]]]:
    if not in_bounds(grid, sx, sy) or not in_bounds(grid, gx, gy):
        return None
    if not is_free(grid, sx, sy) or not is_free(grid, gx, gy):
        return None

    start = (sx, sy)
    goal = (gx, gy)

    open_heap: List[Tuple[int, Tuple[int, int]]] = []  # (f, (x,y))
    heapq.heappush(open_heap, (manhattan(sx, sy, gx, gy), start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g: Dict[Tuple[int, int], int] = {start: 0}

    closed: set[Tuple[int, int]] = set()

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            # 重建路径
            path: List[Tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        cx, cy = current
        for nx, ny in neighbors4(grid, cx, cy):
            tentative_g = g[current] + 1
            if tentative_g < g.get((nx, ny), 1 << 60):
                came_from[(nx, ny)] = current
                g[(nx, ny)] = tentative_g
                fscore = tentative_g + manhattan(nx, ny, gx, gy)
                heapq.heappush(open_heap, (fscore, (nx, ny)))
    return None


def render_with_path(grid: Grid, path: List[Tuple[int, int]]) -> str:
    h, w = len(grid), len(grid[0])
    on_path = set(path)
    lines: List[str] = []
    for y in range(h):
        cells: List[str] = []
        for x in range(w):
            if (x, y) in on_path:
                cells.append("*")
            else:
                cells.append(str(grid[y][x]))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="A* 路径规划（4-邻接，步长=1）")
    parser.add_argument("map_path", type=str, help="地图文件路径 (0=可通行,1=障碍)")
    parser.add_argument("start_x", type=int)
    parser.add_argument("start_y", type=int)
    parser.add_argument("goal_x", type=int)
    parser.add_argument("goal_y", type=int)
    args = parser.parse_args()

    grid = load_map(Path(args.map_path))
    path = astar(grid, args.start_x, args.start_y, args.goal_x, args.goal_y)
    if path is None:
        print(f"I can’t go to the postion ({args.goal_x},{args.goal_y}).")
        return 1
    print(render_with_path(grid, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


