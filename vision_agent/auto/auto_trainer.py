import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoTrainer:
    """自动训练管线：收集帧 → LLM 标注 → 训练模型 → 返回模型路径。

    整个过程不需要人工干预，用于 AutoPilot 检测到新场景时自动触发。
    """

    def __init__(self, output_base_dir: str = "runs/auto"):
        self._output_base_dir = output_base_dir
        self._on_log = None
        self._stop_flag = False

    def set_log_callback(self, callback):
        self._on_log = callback

    def stop(self):
        self._stop_flag = True

    def train_from_video(
        self,
        video_path: str,
        profile_name: str,
        yolo_model: str,
        actions: list[str],
        action_descriptions: dict[str, str] | None = None,
        llm_provider_name: str = "claude",
        llm_api_key: str = "",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_base_url: str = "",
        sample_count: int = 500,
        sample_interval: int = 30,
        model_type: str = "mlp",
        epochs: int = 100,
        use_tool_calling: bool = True,
        progress_callback=None,
    ) -> dict:
        """完整的自动训练流程。

        Returns:
            {"model_dir": str, "annotated_count": int, "accuracy": float, "phases": [...]}
        """
        self._stop_flag = False
        output_dir = Path(self._output_base_dir) / profile_name / time.strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {"model_dir": "", "phases": []}

        # Phase 1: LLM 自动标注
        self._log(f"[Phase 1/2] LLM 自动标注开始 (目标 {sample_count} 帧)")
        if progress_callback:
            progress_callback("annotating", 0.0)

        annotate_path = str(output_dir / "annotated.jsonl")

        from ..core.detector import Detector
        from ..decision.llm_provider import create_provider
        from ..data.auto_annotator import AutoAnnotator

        detector = Detector(model=yolo_model)
        provider = create_provider(llm_provider_name, llm_api_key or "ollama", llm_model, llm_base_url)

        annotator = AutoAnnotator(
            video_path=video_path,
            detector=detector,
            provider=provider,
            actions=actions,
            action_descriptions=action_descriptions,
            sample_interval=sample_interval,
            max_frames=sample_count,
            use_tool_calling=use_tool_calling,
            progress_callback=lambda cur, total, ann: (
                progress_callback("annotating", cur / max(total, 1)) if progress_callback else None
            ),
        )
        if self._on_log:
            annotator.set_log_callback(self._on_log)

        if self._stop_flag:
            return result

        stats = annotator.run(save_path=annotate_path)
        result["annotated_count"] = stats["annotated"]
        result["phases"].append({"phase": "annotate", "stats": stats})
        self._log(f"[Phase 1/2] 标注完成: {stats['annotated']} 条")

        if stats["annotated"] < 10:
            self._log("[终止] 标注数据太少 (<10)，无法训练")
            return result

        if self._stop_flag:
            return result

        # Phase 2: 训练决策模型
        self._log(f"[Phase 2/2] 训练决策模型 ({model_type})")
        if progress_callback:
            progress_callback("training", 0.0)

        model_dir = str(output_dir / "model")

        from ..data.dataset import ActionDataset
        from ..data.train import DecisionTrainer

        dataset = ActionDataset(str(output_dir))
        total = dataset.load()
        self._log(f"加载 {total} 条训练数据")

        trainer = DecisionTrainer(
            dataset=dataset,
            output_dir=model_dir,
            model_type=model_type,
            epochs=epochs,
        )

        metrics = trainer.train(
            progress_callback=lambda epoch, total_ep, loss, t_acc, v_acc: (
                progress_callback("training", epoch / max(total_ep, 1)) if progress_callback else None
            ),
        )

        result["model_dir"] = model_dir
        result["accuracy"] = metrics.get("best_val_acc", 0)
        result["phases"].append({"phase": "train", "metrics": metrics})
        self._log(f"[Phase 2/2] 训练完成: accuracy={result['accuracy']:.3f} → {model_dir}")

        if progress_callback:
            progress_callback("done", 1.0)

        return result

    def train_from_buffer(
        self,
        frames: list,
        profile_name: str,
        yolo_model: str,
        actions: list[str],
        action_descriptions: dict[str, str] | None = None,
        llm_provider_name: str = "claude",
        llm_api_key: str = "",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_base_url: str = "",
        model_type: str = "mlp",
        epochs: int = 100,
        use_tool_calling: bool = True,
        progress_callback=None,
    ) -> dict:
        """从内存帧缓冲区训练（用于实时场景，不需要视频文件）。

        先将帧写为临时视频，再调用 train_from_video。
        """
        import cv2

        if not frames:
            self._log("[终止] 帧缓冲区为空")
            return {"model_dir": "", "phases": []}

        h, w = frames[0].shape[:2]
        tmp_dir = Path(self._output_base_dir) / profile_name / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_video = str(tmp_dir / "buffer.mp4")

        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        self._log(f"帧缓冲区写入临时视频: {len(frames)} 帧 → {tmp_video}")

        return self.train_from_video(
            video_path=tmp_video,
            profile_name=profile_name,
            yolo_model=yolo_model,
            actions=actions,
            action_descriptions=action_descriptions,
            llm_provider_name=llm_provider_name,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            sample_count=len(frames),
            sample_interval=1,
            model_type=model_type,
            epochs=epochs,
            use_tool_calling=use_tool_calling,
            progress_callback=progress_callback,
        )

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
