# -*- coding: utf-8 -*-
"""
机器人行为插件化加载器：
- 通过元类自动注册所有继承自 BaseSkill 的子类。
- SkillManager 自动发现并实例化可用技能。

数学思考：类空间上的“自动注册”相当于对可用策略集合 \(\mathcal{S}\) 的在线增广，
我们将注册表视为 \(\mathcal{S}\) 的显式索引，使得在运行时以次线性复杂度完成查找。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Type


class SkillMeta(type):
    """元类：自动收集子类到注册表。"""

    registry: Dict[str, Type["BaseSkill"]] = {}

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, dict(namespace))
        # 跳过基类自身
        if name != "BaseSkill":
            SkillMeta.registry[name] = cls  # 名称唯一索引
        return cls


class BaseSkill(metaclass=SkillMeta):
    """技能基类：所有插件需继承并实现 run()。"""

    def run(self) -> str:
        raise NotImplementedError


@dataclass
class SkillManager:
    autoload: bool = True

    def list_skills(self) -> List[str]:
        return sorted(SkillMeta.registry.keys())

    def create(self, name: str) -> BaseSkill:
        if name not in SkillMeta.registry:
            raise KeyError(f"未找到技能：{name}")
        return SkillMeta.registry[name]()

    def run_all(self) -> List[str]:
        outputs: List[str] = []
        for name in self.list_skills():
            inst = self.create(name)
            outputs.append(inst.run())
        return outputs


