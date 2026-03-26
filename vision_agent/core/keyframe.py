"""关键帧检测：从视频中智能选取有代表性和高变化的帧。

策略：
  1. 变化检测 — 帧间像素差异大的位置（场景切换、动作爆发）
  2. 均匀保底 — 确保时间线上的均匀覆盖
  3. 去重过滤 — 相似帧只保留一个

相比均匀采样的优势：
  - 团战/技能释放等高变化场景被密集采样
  - 站桩/等待等低变化场景被稀疏采样
  - 同样 300 帧配额，信息密度高数倍
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KeyFrame:
    """关键帧信息。"""
    frame_idx: int       # 原始帧号
    timestamp: float     # 秒
    score: float         # 关键程度（0-1）
    reason: str          # "change" / "uniform" / "scene_cut"
    frame: np.ndarray | None = None  # BGR 图像（可选保留）


class KeyFrameSampler:
    """关键帧采样器。

    用法:
        sampler = KeyFrameSampler(target_count=300)
        keyframes = sampler.sample("video.mp4")
        # keyframes: List[KeyFrame]，已按 frame_idx 排序

    原理:
        1. 快速扫描全视频，计算每帧的变化分数（帧差 + 直方图差）
        2. 选取变化分数最高的 N 帧作为"关键帧"
        3. 补充均匀采样帧确保时间覆盖
        4. 去重（特征距离太近的合并）
    """

    def __init__(
        self,
        target_count: int = 300,
        change_ratio: float = 0.6,
        scan_interval: int = 30,
        min_frame_gap: int = 15,
        on_log=None,
    ):
        """
        Args:
            target_count: 目标关键帧数
            change_ratio: 变化帧占比（0.6 = 60%变化帧 + 40%均匀帧）
            scan_interval: 扫描间隔（每 N 帧计算一次差异）
            min_frame_gap: 最小帧间隔（去重用）
        """
        self._target = target_count
        self._change_ratio = change_ratio
        self._scan_interval = scan_interval
        self._min_gap = min_frame_gap
        self._on_log = on_log

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass

    def sample(self, video_path: str, return_frames: bool = False) -> list[KeyFrame]:
        """从视频中采样关键帧。

        Args:
            video_path: 视频文件路径
            return_frames: 是否在 KeyFrame.frame 中保留图像数据

        Returns:
            按 frame_idx 排序的关键帧列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._log(f"[关键帧] 扫描视频: {total_frames} 帧, 目标 {self._target} 关键帧")

        # ── Phase 1: 快速扫描，计算帧变化分数 ──
        scores = {}  # frame_idx → change_score
        prev_gray = None
        prev_hist = None
        frame_idx = 0

        while True:
            ret = cap.grab()  # grab 不解码，比 read 快很多
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self._scan_interval != 0:
                continue

            ret, frame = cap.retrieve()  # 只在需要时解码
            if not ret:
                break

            # 缩小到 160x90 加速计算
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # 帧差分数
                diff = cv2.absdiff(gray, prev_gray)
                pixel_score = diff.mean() / 255.0

                # 直方图差异
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                hist = hist.flatten() / hist.sum()
                if prev_hist is not None:
                    hist_score = 1.0 - cv2.compareHist(
                        prev_hist.astype(np.float32),
                        hist.astype(np.float32),
                        cv2.HISTCMP_CORREL,
                    )
                else:
                    hist_score = 0.0
                prev_hist = hist

                # 综合分数
                score = 0.7 * pixel_score + 0.3 * hist_score
                scores[frame_idx] = score
            else:
                # 第一帧
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                prev_hist = hist.flatten() / hist.sum()
                scores[frame_idx] = 0.5  # 首帧给中等分数

            prev_gray = gray

        if not scores:
            cap.release()
            return []

        # ── Phase 2: 选取关键帧 ──
        n_change = int(self._target * self._change_ratio)
        n_uniform = self._target - n_change

        # 变化帧：按分数排序取 top-N
        sorted_by_score = sorted(scores.items(), key=lambda x: -x[1])
        change_fids = set()
        for fid, sc in sorted_by_score:
            if len(change_fids) >= n_change:
                break
            # 检查与已选帧的最小间距
            if any(abs(fid - existing) < self._min_gap for existing in change_fids):
                continue
            change_fids.add(fid)

        # 均匀帧：在未被选中的区域均匀补充
        all_fids = sorted(scores.keys())
        uniform_interval = max(1, len(all_fids) // max(n_uniform, 1))
        uniform_fids = set()
        for i in range(0, len(all_fids), uniform_interval):
            fid = all_fids[i]
            if len(uniform_fids) >= n_uniform:
                break
            if fid not in change_fids:
                uniform_fids.add(fid)

        # 合并
        selected_fids = sorted(change_fids | uniform_fids)

        # ── Phase 3: 读取选中帧 ──
        keyframes = []
        fid_set = set(selected_fids)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx not in fid_set:
                continue

            score = scores.get(frame_idx, 0.0)
            reason = "change" if frame_idx in change_fids else "uniform"

            # 场景切换检测（分数特别高）
            if score > 0.3:
                reason = "scene_cut"

            kf = KeyFrame(
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                score=score,
                reason=reason,
                frame=frame.copy() if return_frames else None,
            )
            keyframes.append(kf)

        cap.release()

        # 统计
        n_cuts = sum(1 for kf in keyframes if kf.reason == "scene_cut")
        n_changes = sum(1 for kf in keyframes if kf.reason == "change")
        n_uniforms = sum(1 for kf in keyframes if kf.reason == "uniform")
        self._log(
            f"[关键帧] 采样完成: {len(keyframes)} 帧 "
            f"(场景切换 {n_cuts} + 变化 {n_changes} + 均匀 {n_uniforms})"
        )

        return keyframes

    def sample_indices(self, video_path: str) -> list[int]:
        """只返回关键帧的帧号列表。"""
        keyframes = self.sample(video_path, return_frames=False)
        return [kf.frame_idx for kf in keyframes]
