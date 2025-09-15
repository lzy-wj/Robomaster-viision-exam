# -*- coding: utf-8 -*-
"""
LRU 缓存（最近最少使用）

数学思考：
- 我们维护一个有序集合，表示访问序列的一个“滑动窗口”的最优子结构。
- 每次命中/写入都将元素移至序列前端；淘汰在尾部，等价于近似最小化未来访问的失效率（经验风险意义下）。

实现：使用 OrderedDict 实现 O(1) 的 get/put 与移动操作。
复杂度：get/put 均摊 O(1)，空间 O(C)（C 为容量）。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Hashable, MutableMapping, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("容量必须为正整数")
        self._cap: int = capacity
        self._map: "OrderedDict[K, V]" = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        if key not in self._map:
            return None
        # 移动到最近使用（队首）
        self._map.move_to_end(key, last=False)
        return self._map[key]

    def put(self, key: K, value: V) -> None:
        if key in self._map:
            self._map.move_to_end(key, last=False)
            self._map[key] = value
            return
        # 新插入到队首
        self._map[key] = value
        self._map.move_to_end(key, last=False)
        # 超容量则淘汰队尾（最久未使用）
        if len(self._map) > self._cap:
            self._map.popitem(last=True)

    def __len__(self) -> int:  # 便于测试
        return len(self._map)


