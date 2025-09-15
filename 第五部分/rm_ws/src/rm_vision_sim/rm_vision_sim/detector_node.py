# -*- coding: utf-8 -*-
"""
detector_node：增强版蓝色目标检测节点，支持配置文件、性能监控和质心平滑。

数学思考：
- 颜色分割：在 HSV 空间进行概率密度阈值检验，P(pixel ∈ target | H,S,V)
- 质心估计：像素坐标的最大似然估计 (x̄,ȳ) = argmax Σᵢ wᵢ(xᵢ,yᵢ)
- 指数平滑：xₜ = αxₜ₋₁ + (1-α)x̂ₜ，优化噪声方差与响应延迟的权衡
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_node")
        
        # 声明参数（带默认值）
        self.declare_parameters(
            namespace='',
            parameters=[
                ('hue_lower', 90),
                ('hue_upper', 130),
                ('saturation_lower', 80),
                ('saturation_upper', 255),
                ('value_lower', 50),
                ('value_upper', 255),
                ('morphology_kernel_size', 3),
                ('morphology_iterations', 1),
                ('min_contour_area', 100.0),
                ('max_contour_area', 10000.0),
                ('enable_fps_monitor', True),
                ('fps_report_interval', 30),
                ('enable_centroid_smoothing', True),
                ('smoothing_alpha', 0.7),
                ('enable_debug_image', False),
                ('debug_topic', '/debug/detection_result'),
                # 鲁棒性增强参数
                ('enable_clahe', True),
                ('clahe_clip_limit', 2.0),
                ('clahe_tile_grid', 8),
                ('adaptive_value_norm', True),
                ('use_chromaticity', True),
                ('chromaticity_b_threshold', 0.45),
                ('adaptive_v_percentile', 5),
                ('adaptive_v_margin', 10),
                ('pre_blur_ksize', 3),
            ]
        )
        
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/image_raw", self.on_image, 10)
        self.pub = self.create_publisher(PointStamped, "/target_position", 10)
        
        # 可选的调试图像发布器
        if self.get_parameter('enable_debug_image').value:
            debug_topic = self.get_parameter('debug_topic').value
            self.debug_pub = self.create_publisher(Image, debug_topic, 5)
        else:
            self.debug_pub = None

        # 性能监控
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # 质心平滑状态
        self.smoothed_centroid: Optional[Tuple[float, float]] = None
        
        self.get_logger().info("检测节点已启动，配置已加载")
        self._log_config()

    def _log_config(self) -> None:
        """记录当前配置到日志"""
        config_str = (
            f"HSV阈值: H[{self.get_parameter('hue_lower').value}-{self.get_parameter('hue_upper').value}] "
            f"S[{self.get_parameter('saturation_lower').value}-{self.get_parameter('saturation_upper').value}] "
            f"V[{self.get_parameter('value_lower').value}-{self.get_parameter('value_upper').value}]"
        )
        self.get_logger().info(config_str)

    def _get_hsv_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """根据参数构建HSV阈值"""
        lower = np.array([
            self.get_parameter('hue_lower').value,
            self.get_parameter('saturation_lower').value,
            self.get_parameter('value_lower').value
        ], dtype=np.uint8)
        
        upper = np.array([
            self.get_parameter('hue_upper').value,
            self.get_parameter('saturation_upper').value,
            self.get_parameter('value_upper').value
        ], dtype=np.uint8)
        
        return lower, upper

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """应用形态学操作去噪"""
        kernel_size = self.get_parameter('morphology_kernel_size').value
        iterations = self.get_parameter('morphology_iterations').value
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    def _filter_contours(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """通过轮廓面积过滤并计算质心"""
        # 形心用质心 + 符合圆的鲁棒拟合（可选加权）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = self.get_parameter('min_contour_area').value
        max_area = self.get_parameter('max_contour_area').value
        
        # 找到面积在合理范围内的最大轮廓
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        
        if not valid_contours:
            return None
        
        # 使用最大轮廓计算质心
        largest_contour = max(valid_contours, key=cv2.contourArea)
        m = cv2.moments(largest_contour)
        center_from_moments = None
        if m["m00"] > 0:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            center_from_moments = (cx, cy)

        # 使用最小二乘拟合圆作为补充估计，提高对边缘缺失/噪声的鲁棒性
        (x_fit, y_fit) = (None, None)
        if len(largest_contour) >= 5:
            try:
                (x0, y0), radius = cv2.minEnclosingCircle(largest_contour)
                x_fit, y_fit = float(x0), float(y0)
            except Exception:
                x_fit, y_fit = None, None

        if center_from_moments is not None and x_fit is not None:
            # 融合两个估计，简单平均（也可按面积或置信度加权）
            return 0.5 * (center_from_moments[0] + x_fit), 0.5 * (center_from_moments[1] + y_fit)
        if center_from_moments is not None:
            return center_from_moments
        if x_fit is not None:
            return x_fit, y_fit
        
        return None

    def _smooth_centroid(self, raw_centroid: Tuple[float, float]) -> Tuple[float, float]:
        """指数移动平均平滑质心"""
        if not self.get_parameter('enable_centroid_smoothing').value:
            return raw_centroid
        
        alpha = self.get_parameter('smoothing_alpha').value
        
        if self.smoothed_centroid is None:
            self.smoothed_centroid = raw_centroid
        else:
            cx_smooth = alpha * self.smoothed_centroid[0] + (1 - alpha) * raw_centroid[0]
            cy_smooth = alpha * self.smoothed_centroid[1] + (1 - alpha) * raw_centroid[1]
            self.smoothed_centroid = (cx_smooth, cy_smooth)
        
        return self.smoothed_centroid

    def _update_fps_monitor(self) -> None:
        """更新帧率监控"""
        if not self.get_parameter('enable_fps_monitor').value:
            return
        
        self.frame_count += 1
        interval = self.get_parameter('fps_report_interval').value
        
        if self.frame_count % interval == 0:
            elapsed = time.time() - self.fps_start_time
            fps = interval / elapsed
            self.get_logger().info(f"检测帧率: {fps:.2f} FPS")
            self.fps_start_time = time.time()

    def _publish_debug_image(self, bgr: np.ndarray, mask: np.ndarray, 
                           centroid: Optional[Tuple[float, float]], msg: Image) -> None:
        """发布调试图像（可选）"""
        if self.debug_pub is None:
            return
        
        # 创建三通道调试图像
        debug_img = bgr.copy()
        
        # 叠加掩码（绿色）
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_colored, 0.3, 0)
        
        # 绘制质心十字
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.drawMarker(debug_img, (cx, cy), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.putText(debug_img, f"({cx},{cy})", (cx+10, cy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 发布调试图像
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    def on_image(self, msg: Image) -> None:
        """主处理函数：图像 → 检测 → 发布"""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ksize = int(self.get_parameter('pre_blur_ksize').value)
            if ksize >= 3 and ksize % 2 == 1:
                bgr = cv2.GaussianBlur(bgr, (ksize, ksize), 0)

            # 亮度鲁棒性增强：CLAHE + 可选 V 通道归一化
            if self.get_parameter('enable_clahe').value:
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clip = float(self.get_parameter('clahe_clip_limit').value)
                grid = int(self.get_parameter('clahe_tile_grid').value)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            if self.get_parameter('adaptive_value_norm').value:
                h, s, v = cv2.split(hsv)
                v = cv2.equalizeHist(v)
                hsv = cv2.merge([h, s, v])
            
            # HSV阈值分割
            lower, upper = self._get_hsv_bounds()
            # 自适应调整 V 的下界，降低亮度波动影响
            try:
                v_channel = hsv[:, :, 2]
                perc = int(self.get_parameter('adaptive_v_percentile').value)
                margin = int(self.get_parameter('adaptive_v_margin').value)
                perc_val = float(np.percentile(v_channel, max(0, min(100, perc))))
                dyn_v_lower = max(int(lower[2]), int(max(0, min(255, perc_val - margin))))
                lower = np.array([lower[0], lower[1], dyn_v_lower], dtype=np.uint8)
            except Exception:
                pass

            mask_hsv = cv2.inRange(hsv, lower, upper)

            # 归一化 RGB 色度判别（主要看 B 分量占比），与 HSV 掩码联合
            if self.get_parameter('use_chromaticity').value:
                bgr_f = bgr.astype(np.float32)
                b = bgr_f[:, :, 0]
                g = bgr_f[:, :, 1]
                r = bgr_f[:, :, 2]
                denom = r + g + b + 1e-6
                b_ratio = b / denom
                thr = float(self.get_parameter('chromaticity_b_threshold').value)
                mask_chroma = (b_ratio >= thr).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask_hsv, mask_chroma)
            else:
                mask = mask_hsv
            
            # 形态学去噪
            mask = self._apply_morphology(mask)
            
            # 轮廓过滤与质心计算
            raw_centroid = self._filter_contours(mask)
            
            if raw_centroid is not None:
                # 质心平滑
                smoothed_centroid = self._smooth_centroid(raw_centroid)
                
                # 发布结果
                pt = PointStamped()
                pt.header.stamp = msg.header.stamp
                pt.header.frame_id = "camera_frame"
                pt.point.x = smoothed_centroid[0]
                pt.point.y = smoothed_centroid[1]
                pt.point.z = 0.0  # 图像平面
                self.pub.publish(pt)
                try:
                    # 直接打印，确保终端可见
                    print(f"Target detected at: [x={smoothed_centroid[0]:.1f}, y={smoothed_centroid[1]:.1f}]", flush=True)
                except Exception:
                    pass
                # ROS 日志（可通过过滤控制级别）
                self.get_logger().info(
                    f"Target detected at: [x={smoothed_centroid[0]:.1f}, y={smoothed_centroid[1]:.1f}]"
                )
                
                # 调试图像（可选）
                self._publish_debug_image(bgr, mask, smoothed_centroid, msg)
            
            # 性能监控
            self._update_fps_monitor()
            
        except Exception as e:
            self.get_logger().error(f"图像处理异常: {e}")


def main() -> None:
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


