# -*- coding: utf-8 -*-
"""
Dijkstra 最短路（非负权图）

数学思考：
- 使用贪心选择的可证明性质：每次取最小暂定距离顶点 u，
  则 u 的最短距离被“锁定”（交换论证 + 三角不等式）。
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple


def dijkstra(n: int, edges: List[Tuple[int, int, float]], src: int) -> List[float]:
    graph: Dict[int, List[Tuple[int, float]] ] = {i: [] for i in range(n)}
    for u, v, w in edges:
        if w < 0:
            raise ValueError("Dijkstra 要求非负权重")
        graph[u].append((v, w))

    dist = [float("inf")] * n
    dist[src] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


