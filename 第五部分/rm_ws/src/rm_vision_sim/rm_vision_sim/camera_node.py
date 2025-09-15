# -*- coding: utf-8 -*-
"""
camera_node：模拟相机发布 640x480 黑底 + 移动蓝色圆（半径=20），发布频率 10Hz。

数学思考：
- 目标运动建模为离散时间系统 x_{k+1} = x_k + v_k Δt + w_k，
  这里用小随机项 w_k（均匀噪声）模拟缓慢随机移动，保证状态在边界处反弹保持可观测性。
"""

from __future__ import annotations

import random
import time
from typing import Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")
        self.publisher_ = self.create_publisher(Image, "/image_raw", 10)
        self.gt_pub = self.create_publisher(PointStamped, "/target_gt", 10)
        self.bridge = CvBridge()

        self.height = 480
        self.width = 640
        self.radius = 20
        # 初始位置置于图像中心
        self.pos_x = self.width // 2
        self.pos_y = self.height // 2
        # 缓慢移动的速度分量（像素/帧）
        self.vx = 1
        self.vy = 1

        self.timer = self.create_timer(0.1, self.on_timer)  # 10 Hz
        self.start_time = time.time()
        self.flicker_enabled = True  # 固定开启：30s 后开始交替变亮/变暗

    def _step(self) -> None:
        # 轻微随机扰动，模拟非完全匀速
        self.vx += random.choice([-1, 0, 1]) * 0.1
        self.vy += random.choice([-1, 0, 1]) * 0.1
        self.vx = max(-2.0, min(2.0, self.vx))
        self.vy = max(-2.0, min(2.0, self.vy))

        self.pos_x = int(self.pos_x + self.vx)
        self.pos_y = int(self.pos_y + self.vy)

        # 边界反弹
        if self.pos_x - self.radius < 0:
            self.pos_x = self.radius
            self.vx = abs(self.vx)
        if self.pos_x + self.radius >= self.width:
            self.pos_x = self.width - self.radius - 1
            self.vx = -abs(self.vx)
        if self.pos_y - self.radius < 0:
            self.pos_y = self.radius
            self.vy = abs(self.vy)
        if self.pos_y + self.radius >= self.height:
            self.pos_y = self.height - self.radius - 1
            self.vy = -abs(self.vy)

    def _render(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # BGR 蓝色 (255, 0, 0)
        cv2.circle(img, (self.pos_x, self.pos_y), self.radius, (255, 0, 0), thickness=-1)
        # 30s 后引入亮灭扰动：每 1s 在明/暗之间切换
        if self.flicker_enabled:
            elapsed = time.time() - self.start_time
            if elapsed >= 30.0:
                phase = int((elapsed - 30.0)) % 2  # 0,1 交替
                if phase == 0:
                    # 变亮：整体提升亮度（加上常量并裁剪）
                    img = cv2.convertScaleAbs(img, alpha=1.0, beta=80)
                else:
                    # 变暗：整体降低亮度
                    img = cv2.convertScaleAbs(img, alpha=1.0, beta=-60)
        return img

    def on_timer(self) -> None:
        self._step()
        frame = self._render()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)
        # 发布真值圆心（用于评测与可视化误差）
        gt = PointStamped()
        gt.header.stamp = msg.header.stamp
        gt.header.frame_id = "camera_frame"
        gt.point.x = float(self.pos_x)
        gt.point.y = float(self.pos_y)
        gt.point.z = 0.0
        self.gt_pub.publish(gt)


def main() -> None:
    rclpy.init()
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


