"""从帧的固定区域提取特征信息。"""

import cv2
import numpy as np


class ROIExtractor:
    """从帧的固定区域提取特征信息（血条、蓝条、数字区域等）。"""

    def __init__(self, regions: dict[str, tuple] | None = None):
        """
        Args:
            regions: {"hp_bar": (x1_norm, y1_norm, x2_norm, y2_norm), ...}
                     归一化坐标 [0, 1]
        """
        self._regions = regions or {}

    def set_regions(self, regions: dict[str, tuple]):
        self._regions = regions

    def extract(self, frame: np.ndarray) -> dict[str, dict]:
        """从帧中提取所有 ROI 的特征。"""
        results = {}
        for name, region in self._regions.items():
            crop = self._crop_region(frame, region)
            if crop.size == 0:
                continue
            color_info = self._analyze_color(crop)
            results[name] = {
                "crop": crop,
                "mean_color": color_info["mean_color"],
                "color_ratio": color_info["color_ratio"],
                "brightness": color_info["brightness"],
            }
        return results

    def extract_bar_ratio(self, frame: np.ndarray, region_name: str,
                          bar_color: tuple = (0, 255, 0)) -> float:
        """提取条状 UI 的填充比例。在 HSV 空间做颜色范围匹配。"""
        if region_name not in self._regions:
            return 0.0

        crop = self._crop_region(frame, self._regions[region_name])
        if crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # BGR -> HSV 参考色，构建容差范围
        ref_bgr = np.uint8([[list(bar_color)]])
        ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h_tol, s_tol, v_tol = 15, 60, 60
        lower = np.array([
            max(0, int(ref_hsv[0]) - h_tol),
            max(0, int(ref_hsv[1]) - s_tol),
            max(0, int(ref_hsv[2]) - v_tol),
        ])
        upper = np.array([
            min(179, int(ref_hsv[0]) + h_tol),
            255,
            255,
        ])

        mask = cv2.inRange(hsv, lower, upper)

        # 按列统计：每列是否有匹配像素
        col_has_bar = np.any(mask > 0, axis=0)
        if len(col_has_bar) == 0:
            return 0.0

        # 从左到右找最后一个有填充的列
        filled_cols = np.where(col_has_bar)[0]
        if len(filled_cols) == 0:
            return 0.0

        rightmost = filled_cols[-1] + 1
        return rightmost / len(col_has_bar)

    def extract_number_region(self, frame: np.ndarray, region_name: str) -> np.ndarray:
        """提取包含数字的 ROI，做灰度 + 二值化 + 形态学处理，供 OCR 使用。"""
        if region_name not in self._regions:
            return np.array([])

        crop = self._crop_region(frame, self._regions[region_name])
        if crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
        # 形态学闭操作，连接断裂笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return processed

    def _crop_region(self, frame: np.ndarray, region: tuple) -> np.ndarray:
        """根据归一化坐标裁剪区域。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    def _analyze_color(self, crop: np.ndarray) -> dict:
        """分析颜色分布。"""
        mean_bgr = cv2.mean(crop)[:3]
        b, g, r = mean_bgr
        total = b + g + r + 1e-8

        # 亮度：转灰度取均值再归一化到 [0, 1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0

        return {
            "mean_color": (round(b, 1), round(g, 1), round(r, 1)),
            "color_ratio": {
                "blue": round(b / total, 3),
                "green": round(g / total, 3),
                "red": round(r / total, 3),
            },
            "brightness": round(brightness, 3),
        }
