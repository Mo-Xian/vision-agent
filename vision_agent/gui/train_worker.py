"""训练工作线程。"""

from PySide6.QtCore import QThread, Signal
from ..core.trainer import Trainer, TrainConfig


class TrainWorker(QThread):
    """后台训练线程。"""

    log_message = Signal(str)       # 日志消息
    finished_ok = Signal(str)       # 训练完成，参数为最佳模型路径
    finished_err = Signal(str)      # 训练失败，参数为错误信息

    def __init__(self, config: TrainConfig, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            self.log_message.emit(f"加载基础模型: {self.config.base_model}")
            self.log_message.emit(f"数据集: {self.config.data_yaml}")
            self.log_message.emit(f"训练轮数: {self.config.epochs}, 批大小: {self.config.batch}")
            self.log_message.emit("开始训练...")

            trainer = Trainer(self.config)
            best_path = trainer.train()

            self.log_message.emit(f"训练完成! 最佳模型: {best_path}")

            # 验证
            try:
                metrics = trainer.validate(best_path)
                self.log_message.emit(
                    f"验证结果: mAP50={metrics['mAP50']:.4f}, "
                    f"mAP50-95={metrics['mAP50-95']:.4f}, "
                    f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
                )
            except Exception as e:
                self.log_message.emit(f"验证跳过: {e}")

            self.finished_ok.emit(best_path)
        except Exception as e:
            self.log_message.emit(f"训练失败: {e}")
            self.finished_err.emit(str(e))
