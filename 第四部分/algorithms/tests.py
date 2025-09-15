# -*- coding: utf-8 -*-
from __future__ import annotations

from lru_cache import LRUCache
from union_find import UnionFind
from dijkstra import dijkstra


def test_lru() -> None:
    lru = LRUCache[int, int](capacity=2)
    lru.put(1, 1)
    lru.put(2, 2)
    assert lru.get(1) == 1  # 访问 1
    lru.put(3, 3)  # 淘汰 2
    assert lru.get(2) is None
    assert lru.get(1) == 1
    assert lru.get(3) == 3


def test_union_find() -> None:
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(3, 4)
    assert uf.connected(1, 2)
    assert not uf.connected(1, 3)
    uf.union(2, 3)
    assert uf.connected(1, 4)


def test_dijkstra() -> None:
    n = 5
    edges = [
        (0, 1, 2.0),
        (0, 2, 4.0),
        (1, 2, 1.0),
        (1, 3, 7.0),
        (2, 4, 3.0),
        (4, 3, 2.0),
    ]
    dist = dijkstra(n, edges, src=0)
    assert dist[0] == 0.0
    assert abs(dist[3] - 8.0) < 1e-9  # 0->1(2),1->2(1),2->4(3),4->3(2)


def main() -> None:
    test_lru()
    test_union_find()
    test_dijkstra()
    print("OK: algorithms tests passed")


if __name__ == "__main__":
    main()


