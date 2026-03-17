"""决策模型训练工作线程。"""

from PySide6.QtCore import QThread, Signal


class DecisionTrainWorker(QThread):
    """后台训练决策模型（MLP / RandomForest）。"""

    log_message = Signal(str)
    progress = Signal(int, int, float, float, float)  # epoch, total, loss, train_acc, val_acc
    finished_ok = Signal(str, dict)  # (model_dir, metrics)
    finished_err = Signal(str)

    def __init__(self, data_dir: str, output_dir: str, model_type: str = "mlp",
                 hidden_dims: list[int] | None = None, epochs: int = 100,
                 batch_size: int = 64, lr: float = 1e-3, patience: int = 15,
                 dropout: float = 0.3, val_split: float = 0.2,
                 min_detections: int = 0, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_type = model_type
        self.hidden_dims = hidden_dims or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.val_split = val_split
        self.min_detections = min_detections

    def run(self):
        try:
            from ..data.train import DecisionTrainer

            def on_progress(epoch, total, loss, train_acc, val_acc):
                self.progress.emit(epoch, total, loss, train_acc, val_acc)

            self.log_message.emit(f"数据目录: {self.data_dir}")
            self.log_message.emit(f"模型类型: {self.model_type}")
            if self.model_type == "mlp":
                self.log_message.emit(f"网络结构: {self.hidden_dims}, epochs={self.epochs}, lr={self.lr}")
            self.log_message.emit("开始训练...")

            trainer = DecisionTrainer(
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                model_type=self.model_type,
                hidden_dims=self.hidden_dims,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                patience=self.patience,
                dropout=self.dropout,
                val_split=self.val_split,
                min_detections=self.min_detections,
                progress_callback=on_progress,
            )

            metrics = trainer.run()

            self.log_message.emit(
                f"训练完成! val_acc={metrics['best_val_acc']:.4f} "
                f"train_acc={metrics['final_train_acc']:.4f}"
            )
            self.finished_ok.emit(self.output_dir, metrics)

        except Exception as e:
            self.log_message.emit(f"训练失败: {e}")
            self.finished_err.emit(str(e))
