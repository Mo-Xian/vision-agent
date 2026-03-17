"""自动标注工作线程。"""

from PySide6.QtCore import QThread, Signal


class AnnotateWorker(QThread):
    """后台运行 LLM 自动标注。"""

    log_message = Signal(str)
    progress = Signal(int, int, int)  # (current_frame, total_frames, annotated)
    finished_ok = Signal(dict)        # stats
    finished_err = Signal(str)

    def __init__(
        self,
        video_path: str,
        model_path: str,
        provider_name: str,
        api_key: str,
        llm_model: str,
        base_url: str,
        actions: list[str],
        save_path: str,
        sample_interval: int = 10,
        max_frames: int = 0,
        confidence: float = 0.5,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self.provider_name = provider_name
        self.api_key = api_key
        self.llm_model = llm_model
        self.base_url = base_url
        self.actions = actions
        self.save_path = save_path
        self.sample_interval = sample_interval
        self.max_frames = max_frames
        self.confidence = confidence
        self._annotator = None

    def run(self):
        try:
            from ..core.detector import Detector
            from ..decision.llm_provider import create_provider
            from ..data.auto_annotator import AutoAnnotator

            self.log_message.emit(f"加载检测模型: {self.model_path}")
            detector = Detector(model=self.model_path, confidence=self.confidence)

            self.log_message.emit(f"连接 LLM: {self.provider_name}/{self.llm_model}")
            provider = create_provider(
                self.provider_name, self.api_key, self.llm_model, self.base_url,
            )

            self.log_message.emit(f"动作空间: {self.actions}")
            self.log_message.emit(f"采样间隔: 每 {self.sample_interval} 帧")
            if self.max_frames > 0:
                self.log_message.emit(f"最大帧数: {self.max_frames}")
            self.log_message.emit("开始自动标注...")

            def on_progress(current, total, annotated):
                self.progress.emit(current, total, annotated)

            self._annotator = AutoAnnotator(
                video_path=self.video_path,
                detector=detector,
                provider=provider,
                actions=self.actions,
                sample_interval=self.sample_interval,
                max_frames=self.max_frames,
                progress_callback=on_progress,
            )
            self._annotator.set_log_callback(lambda msg: self.log_message.emit(msg))

            stats = self._annotator.run(save_path=self.save_path)

            self.log_message.emit(
                f"标注完成! 采样 {stats['sampled']} 帧, "
                f"标注 {stats['annotated']} 条, "
                f"跳过 {stats['skipped']}, 错误 {stats['errors']}"
            )
            self.log_message.emit(f"动作分布: {stats['action_dist']}")
            self.log_message.emit(f"保存到: {self.save_path}")
            self.finished_ok.emit(stats)

        except Exception as e:
            self.log_message.emit(f"标注失败: {e}")
            self.finished_err.emit(str(e))

    def stop(self):
        if self._annotator:
            self._annotator.stop()
        self.wait(5000)
