#!/usr/bin/env python3
"""
生成大型测试日志文件
用于替代原本的robot.zip大文件
"""

import os
import gzip
import random
import datetime
from pathlib import Path

def generate_large_log_file(output_path: str = "robot_test.log.gz", size_mb: int = 50):
    """
    生成指定大小的压缩日志文件
    
    Args:
        output_path: 输出文件路径
        size_mb: 目标文件大小(MB)
    """
    print(f"🚀 开始生成测试日志文件: {output_path}")
    print(f"📊 目标大小: {size_mb}MB")
    
    # 日志模板
    log_templates = [
        "INFO: Robot position updated: x={x:.2f}, y={y:.2f}, theta={theta:.3f}",
        "DEBUG: Sensor reading - distance: {dist:.1f}m, angle: {angle:.1f}°", 
        "WARN: Battery level low: {battery}%",
        "ERROR: Communication timeout with device {device_id}",
        "INFO: Navigation waypoint reached: ({wp_x:.1f}, {wp_y:.1f})",
        "DEBUG: Motor speed - left: {left_speed}rpm, right: {right_speed}rpm",
        "INFO: Object detected at distance {obj_dist:.2f}m",
        "TRACE: Memory usage: {memory:.1f}MB, CPU: {cpu:.1f}%"
    ]
    
    target_bytes = size_mb * 1024 * 1024
    current_bytes = 0
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        line_count = 0
        
        while current_bytes < target_bytes:
            # 生成时间戳
            timestamp = datetime.datetime.now() + datetime.timedelta(milliseconds=line_count)
            
            # 随机选择日志模板
            template = random.choice(log_templates)
            
            # 生成随机参数
            log_data = {
                'x': random.uniform(-10, 10),
                'y': random.uniform(-10, 10), 
                'theta': random.uniform(0, 6.28),
                'dist': random.uniform(0.1, 5.0),
                'angle': random.uniform(-180, 180),
                'battery': random.randint(10, 100),
                'device_id': random.randint(1, 10),
                'wp_x': random.uniform(-20, 20),
                'wp_y': random.uniform(-20, 20),
                'left_speed': random.randint(0, 3000),
                'right_speed': random.randint(0, 3000),
                'obj_dist': random.uniform(0.1, 3.0),
                'memory': random.uniform(100, 800),
                'cpu': random.uniform(10, 95)
            }
            
            # 格式化日志行
            log_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {template.format(**log_data)}\n"
            
            # 写入文件
            f.write(log_line)
            current_bytes += len(log_line.encode('utf-8'))
            line_count += 1
            
            # 进度显示
            if line_count % 10000 == 0:
                progress = (current_bytes / target_bytes) * 100
                print(f"📝 已生成 {line_count:,} 行日志 ({progress:.1f}%)")
    
    # 检查实际文件大小
    actual_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ 日志文件生成完成!")
    print(f"📄 文件路径: {os.path.abspath(output_path)}")
    print(f"📊 实际大小: {actual_size:.2f}MB")
    print(f"📝 总行数: {line_count:,}")

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 机器人日志测试数据生成器")
    print("=" * 50)
    
    # 确保在正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 生成测试文件
    generate_large_log_file("robot_test.log.gz", 30)  # 30MB压缩文件
    
    print("\n💡 使用提示:")
    print("- 在stream_count_error.py中将文件名改为'robot_test.log.gz'")
    print("- 或者解压缩: gunzip robot_test.log.gz")
    print("- 此文件可安全删除并重新生成")
