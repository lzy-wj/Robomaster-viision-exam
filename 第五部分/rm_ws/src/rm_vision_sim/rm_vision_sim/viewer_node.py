# -*- coding: utf-8 -*-
"""viewer_node：可视化 /image_raw，叠加瞄准十字与检测坐标。

参数：
- window_name: 窗口名（默认 'RM Viewer'）
- crosshair_color: 瞄准十字颜色 BGR（默认 [0,255,0]）
- show_fps: 是否显示帧率（默认 True）
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image


class ViewerNode(Node):
    def __init__(self) -> None:
        super().__init__('viewer_node')

        self.declare_parameters('', [
            ('window_name', 'RM Viewer'),
            ('crosshair_color', [0, 255, 0]),
            ('show_fps', True),
        ])

        self.bridge = CvBridge()
        self.latest_point: Optional[PointStamped] = None
        self.last_time = time.time()
        self.frame_counter = 0
        self.fps = 0.0

        self.sub_img = self.create_subscription(Image, '/image_raw', self.on_image, 10)
        self.sub_pt = self.create_subscription(PointStamped, '/target_position', self.on_point, 10)
        self.sub_gt = self.create_subscription(PointStamped, '/target_gt', self.on_gt, 10)

        self.window_name = str(self.get_parameter('window_name').value)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(0.03, self.on_timer)  # ~33ms, 驱动窗口刷新

    def on_point(self, msg: PointStamped) -> None:
        self.latest_point = msg

    def on_gt(self, msg: PointStamped) -> None:
        # 保存最新的真值，复用 latest_point 的 header 时间轴
        self.latest_gt = msg

    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        # 画中心十字
        color = self.get_parameter('crosshair_color').value
        cx, cy = w // 2, h // 2
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 24, 2)

        # 绘制最新目标点
        if self.latest_point is not None:
            px = int(self.latest_point.point.x)
            py = int(self.latest_point.point.y)
            cv2.circle(out, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(out, f"({px},{py})", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 误差：与真值圆心（/target_gt）比较；若无真值则与图像中心比较
            if hasattr(self, 'latest_gt') and self.latest_gt is not None:
                gx, gy = int(self.latest_gt.point.x), int(self.latest_gt.point.y)
                cv2.circle(out, (gx, gy), 6, (255, 0, 0), 2)
                cv2.line(out, (gx, gy), (px, py), (0, 255, 255), 2)
                err = float(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)
                cv2.putText(out, f"err: {err:.1f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.line(out, (cx, cy), (px, py), (0, 255, 255), 2)
                err = float(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
                cv2.putText(out, f"err: {err:.1f}px", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示 FPS
        if self.get_parameter('show_fps').value:
            cv2.putText(out, f"FPS: {self.fps:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return out

    def on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            disp = self._overlay(frame)
            cv2.imshow(self.window_name, disp)
            cv2.waitKey(1)

            # 更新 FPS 估计
            self.frame_counter += 1
            now = time.time()
            if self.frame_counter % 20 == 0:
                dt = now - self.last_time
                if dt > 0:
                    self.fps = 20.0 / dt
                self.last_time = now
        except Exception as e:
            self.get_logger().error(f'Viewer 处理异常: {e}')

    def on_timer(self) -> None:
        # 保持窗口响应
        cv2.waitKey(1)


def main() -> None:
    rclpy.init()
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



