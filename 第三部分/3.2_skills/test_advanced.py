# -*- coding: utf-8 -*-
"""
manager.py 的高级测试：验证元类注册、动态加载和错误处理机制。

数学思考：
- 类注册可建模为动态集合 S(t)，随类定义事件单调增长
- 测试覆盖率可量化为 |tested_cases| / |total_cases|，确保关键路径验证
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List

# 添加当前目录到Python路径以导入待测模块
sys.path.insert(0, str(Path(__file__).parent))

from manager import BaseSkill, SkillManager, SkillMeta


class TestSkillA(BaseSkill):
    """测试技能A：基础功能验证"""
    def run(self) -> str:
        return "skill_a_executed"


class TestSkillB(BaseSkill):
    """测试技能B：参数化输出"""
    def __init__(self, prefix: str = "default") -> None:
        self.prefix = prefix
    
    def run(self) -> str:
        return f"{self.prefix}_skill_b"


class TestSkillC(BaseSkill):
    """测试技能C：异常场景"""
    def run(self) -> str:
        raise RuntimeError("skill_c_simulated_error")


def test_meta_registration() -> None:
    """测试元类自动注册机制"""
    print("[测试] 元类自动注册")
    
    # 验证已注册的测试技能
    registry = SkillMeta.registry
    assert "TestSkillA" in registry
    assert "TestSkillB" in registry  
    assert "TestSkillC" in registry
    assert "BaseSkill" not in registry  # 基类应被排除
    
    print(f"  已注册技能: {sorted(registry.keys())}")
    print("  ✓ 元类注册机制正常")


def test_manager_basic_operations() -> None:
    """测试管理器基础操作"""
    print("[测试] 管理器基础功能")
    
    manager = SkillManager()
    
    # 技能列表
    skills = manager.list_skills()
    assert "TestSkillA" in skills
    assert "TestSkillB" in skills
    
    # 技能创建
    skill_a = manager.create("TestSkillA")
    assert isinstance(skill_a, TestSkillA)
    result = skill_a.run()
    assert result == "skill_a_executed"
    
    print("  ✓ 基础操作正常")


def test_error_handling() -> None:
    """测试错误处理与异常传播"""
    print("[测试] 错误处理机制")
    
    manager = SkillManager()
    
    # 不存在的技能
    try:
        manager.create("NonExistentSkill")
        assert False, "应抛出KeyError"
    except KeyError as e:
        assert "未找到技能" in str(e)
    
    # 技能执行异常
    skill_c = manager.create("TestSkillC")
    try:
        skill_c.run()
        assert False, "应抛出RuntimeError"
    except RuntimeError as e:
        assert "skill_c_simulated_error" in str(e)
    
    print("  ✓ 错误处理正常")


def test_dynamic_loading() -> None:
    """测试动态技能加载（模拟插件机制）"""
    print("[测试] 动态加载机制")
    
    # 创建临时技能文件
    skill_code = textwrap.dedent("""
        from manager import BaseSkill
        
        class DynamicSkill(BaseSkill):
            def run(self) -> str:
                return "dynamic_skill_loaded"
    """)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(skill_code)
        temp_file = Path(f.name)
    
    try:
        # 动态导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_skill", temp_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        
        # 验证自动注册
        manager = SkillManager()
        assert "DynamicSkill" in manager.list_skills()
        
        # 执行动态技能
        dynamic_skill = manager.create("DynamicSkill")
        result = dynamic_skill.run()
        assert result == "dynamic_skill_loaded"
        
        print("  ✓ 动态加载正常")
        
    finally:
        temp_file.unlink()  # 清理临时文件


def test_batch_execution() -> None:
    """测试批量执行与结果聚合"""
    print("[测试] 批量执行机制")
    
    manager = SkillManager()
    
    # 为了测试，临时移除会抛异常的技能
    original_registry = SkillMeta.registry.copy()
    if "TestSkillC" in SkillMeta.registry:
        del SkillMeta.registry["TestSkillC"]
    
    try:
        outputs = manager.run_all()
        assert len(outputs) >= 2  # 至少有TestSkillA和TestSkillB
        assert "skill_a_executed" in outputs
        
        print(f"  批量执行结果: {outputs}")
        print("  ✓ 批量执行正常")
        
    finally:
        # 恢复注册表
        SkillMeta.registry.clear()
        SkillMeta.registry.update(original_registry)


def test_concurrency_safety() -> None:
    """测试并发安全性（简单验证）"""
    print("[测试] 并发安全性")
    
    import threading
    import time
    
    results: List[str] = []
    errors: List[Exception] = []
    
    def worker(worker_id: int) -> None:
        try:
            manager = SkillManager()
            skill = manager.create("TestSkillA")
            result = skill.run()
            results.append(f"worker_{worker_id}:{result}")
            time.sleep(0.01)  # 模拟处理时间
        except Exception as e:
            errors.append(e)
    
    # 启动多个工作线程
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0, f"并发执行出现错误: {errors}"
    assert len(results) == 5
    
    print(f"  并发结果: {len(results)} 个成功")
    print("  ✓ 并发安全性验证通过")


def main() -> None:
    """运行所有测试"""
    print("=== 高级技能管理器测试 ===\n")
    
    test_functions = [
        test_meta_registration,
        test_manager_basic_operations, 
        test_error_handling,
        test_dynamic_loading,
        test_batch_execution,
        test_concurrency_safety,
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            print()
    
    print(f"=== 测试完成: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("🎉 所有测试通过！技能管理器功能正常。")
        return 0
    else:
        print("❌ 部分测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
