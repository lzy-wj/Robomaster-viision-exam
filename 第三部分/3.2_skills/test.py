# -*- coding: utf-8 -*-
"""测试：在同一文件中定义插件类，验证自动注册与调用。"""

from __future__ import annotations

from manager import BaseSkill, SkillManager


class CruiseSkill(BaseSkill):
    def run(self) -> str:
        return "巡航：沿预设路径移动"


class GraspSkill(BaseSkill):
    def run(self) -> str:
        return "抓取：完成目标抓取动作"


class AvoidSkill(BaseSkill):
    def run(self) -> str:
        return "规避：绕开障碍物"


def main() -> None:
    mgr = SkillManager()
    skills = mgr.list_skills()
    print("skills:", ", ".join(skills))
    outputs = mgr.run_all()
    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()


