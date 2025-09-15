# -*- coding: utf-8 -*-
"""logger_node：订阅目标位置并打印。"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


class LoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("logger_node")
        self.sub = self.create_subscription(PointStamped, "/target_position", self.on_point, 10)

    def on_point(self, msg: PointStamped) -> None:
        text = f"Target detected at: [x={msg.point.x:.1f}, y={msg.point.y:.1f}]"
        # ROS 日志
        self.get_logger().info(text)
        # 直接标准输出，确保终端可见（无缓冲）
        try:
            print(text, flush=True)
        except Exception:
            pass


def main() -> None:
    rclpy.init()
    node = LoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


