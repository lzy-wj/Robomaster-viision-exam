# -*- coding: utf-8 -*-
"""
并查集（Disjoint Set Union, DSU）

数学思考：
- 维护集合划分 \(\{S_i\}\)，支持等价关系的在线合并；
- 带路径压缩与按秩合并使摊还复杂度接近 \(\alpha(n)\)（反阿克曼函数）。
"""

from __future__ import annotations

from typing import Dict


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 按秩合并
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


