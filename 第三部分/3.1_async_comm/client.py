# -*- coding: utf-8 -*-
"""
异步设备通信框架示例：
- 使用 asyncio 并发连接多个 TCP 端点（host:port）。
- 体现数学思考：并发连接可理解为对若干独立“成功事件”的并行估计，
  我们通过超时作为“删失”观测，最小化期望等待时间的上界（超时截断）。
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Target:
    host: str
    port: int

    @classmethod
    def parse_many(cls, text: str) -> List["Target"]:
        targets: List[Target] = []
        if not text:
            return targets
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"无效目标（缺少冒号）: {item}")
            host, port_s = item.rsplit(":", 1)
            targets.append(cls(host=host, port=int(port_s)))
        return targets


async def connect_once(target: Target, timeout: float) -> Tuple[Target, bool, str]:
    """尝试建立一次 TCP 连接并立即关闭。

    数学视角：给定连接时延随机变量 T，设置超时 τ，即观测到 min(T, τ)，
    用以控制最坏情况导致的总等待时间。这里以超时作为删失点。
    
    错误分类：
    - TimeoutError: 超时删失，可能由网络延迟或目标负载过高引起
    - ConnectionRefusedError: 端口关闭或服务未启动  
    - OSError: 网络不可达、DNS解析失败等系统级错误
    """
    reader = None
    writer = None
    
    try:
        # 建立连接（带超时控制）
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(target.host, target.port), timeout=timeout
        )
        
        # 发送轻量探活消息
        message = f"ping {target.host}:{target.port}\n"
        writer.write(message.encode("utf-8"))
        
        # 确保数据发送完成（防止缓冲区延迟）
        await asyncio.wait_for(writer.drain(), timeout=min(timeout * 0.5, 1.0))
        
        return target, True, "connected"
        
    except asyncio.TimeoutError:
        return target, False, "timeout"
    except ConnectionRefusedError:
        return target, False, "connection_refused"
    except OSError as e:
        # 网络层错误（DNS解析失败、网络不可达等）
        if "Name or service not known" in str(e):
            return target, False, "dns_error"
        elif "Network is unreachable" in str(e):
            return target, False, "network_unreachable"
        else:
            return target, False, f"os_error: {e}"
    except Exception as e:
        # 其他未预期的异常
        return target, False, f"unexpected_error: {type(e).__name__}: {e}"
    finally:
        # 安全关闭连接（防止资源泄漏）
        if writer is not None:
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except Exception:
                # 关闭过程中的异常不应影响主流程
                pass


async def run_all(targets: List[Target], timeout: float) -> int:
    """并发测试所有目标的连通性。
    
    数学思考：通过并行化最小化总等待时间 E[T_total] = max(T_i) 而非 Σ T_i，
    在独立连接假设下可获得 O(n) 倍的性能提升。
    """
    if not targets:
        print("WARNING: 没有可测试的目标")
        return 2
    
    print(f"[信息] 开始并发测试 {len(targets)} 个目标，超时={timeout}s")
    
    # 创建并发任务
    tasks = [connect_once(t, timeout) for t in targets]
    
    try:
        # 等待所有任务完成（不抛出异常，收集所有结果）
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"ERROR: 批量任务执行异常: {e}")
        return 3
    
    # 统计结果
    success_count = 0
    error_types = {}
    
    for i, result in enumerate(results):
        target = targets[i]
        
        if isinstance(result, Exception):
            # 任务本身抛出的异常（不应发生，因为connect_once已捕获所有异常）
            print(f"EXCEPTION {target.host}:{target.port} - {result}")
            error_types["task_exception"] = error_types.get("task_exception", 0) + 1
        else:
            # 正常的连接结果
            t, success, info = result
            if success:
                success_count += 1
                print(f"CONNECTED {t.host}:{t.port} - {info}")
            else:
                print(f"FAILED    {t.host}:{t.port} - {info}")
                # 统计错误类型
                error_type = info.split(":")[0] if ":" in info else info
                error_types[error_type] = error_types.get(error_type, 0) + 1
    
    # 输出统计摘要
    print(f"SUMMARY: {success_count}/{len(targets)} connected")
    
    if error_types:
        print("错误统计:", end=" ")
        error_stats = [f"{k}={v}" for k, v in error_types.items()]
        print(", ".join(error_stats))
    
    # 返回码：0=全部成功，1=部分失败，2=输入错误，3=系统错误
    if success_count == len(targets):
        return 0
    elif success_count > 0:
        return 1
    else:
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="并发连接多个 TCP 端点")
    parser.add_argument(
        "--targets",
        type=str,
        default="127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003",
        help="以逗号分隔的 host:port 列表",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.8,
        help="单个连接的超时时间（秒）",
    )
    args = parser.parse_args()

    targets = Target.parse_many(args.targets)
    if not targets:
        print("未提供任何目标端点。")
        return 2

    return asyncio.run(run_all(targets, timeout=args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())


