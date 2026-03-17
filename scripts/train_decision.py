"""决策模型训练 CLI。

用法:
    # MLP 训练 (默认)
    python scripts/train_decision.py --data data/recordings --output runs/decision/exp1

    # RandomForest 快速验证
    python scripts/train_decision.py --data data/recordings --model rf

    # 自定义参数
    python scripts/train_decision.py --data data/recordings --epochs 200 --lr 0.001 --hidden 256 128 64

    # 仅查看数据概况
    python scripts/train_decision.py --data data/recordings --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_agent.data.dataset import ActionDataset
from vision_agent.data.train import DecisionTrainer


def main():
    parser = argparse.ArgumentParser(description="训练决策模型")
    parser.add_argument("--data", required=True, help="录制数据目录")
    parser.add_argument("--output", default="runs/decision/exp1", help="输出目录")
    parser.add_argument("--model", choices=["mlp", "rf"], default="mlp", help="模型类型")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64], help="MLP 隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--min-detections", type=int, default=0, help="最少检测目标数")
    parser.add_argument("--n-estimators", type=int, default=100, help="RF 树数量")
    parser.add_argument("--dry-run", action="store_true", help="仅查看数据概况，不训练")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # dry-run: 仅查看数据
    if args.dry_run:
        dataset = ActionDataset(args.data)
        total = dataset.load()
        if total == 0:
            print("未找到数据。请先用 DataRecorder 录制训练数据。")
            return

        summary = dataset.summary()
        print("\n=== 数据概况 ===")
        print(f"  总样本数: {summary['total']}")
        print(f"\n  动作分布:")
        for action, count in summary["action_distribution"].items():
            pct = count / summary["total"] * 100
            print(f"    {action}: {count} ({pct:.1f}%)")
        print(f"\n  检测类别:")
        for cls, count in summary["detection_classes"].items():
            print(f"    {cls}: {count}")
        return

    # 训练
    def on_progress(epoch, total, loss, train_acc, val_acc):
        bar_len = 30
        filled = int(bar_len * epoch / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {epoch}/{total} "
            f"loss={loss:.4f} train={train_acc:.3f} val={val_acc:.3f}",
            end="", flush=True,
        )
        if epoch == total or (epoch > 1 and val_acc >= 0.99):
            print()

    trainer = DecisionTrainer(
        data_dir=args.data,
        output_dir=args.output,
        model_type=args.model,
        val_split=args.val_split,
        min_detections=args.min_detections,
        hidden_dims=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dropout=args.dropout,
        n_estimators=args.n_estimators,
        progress_callback=on_progress,
    )

    print(f"\n训练配置:")
    print(f"  数据目录: {args.data}")
    print(f"  输出目录: {args.output}")
    print(f"  模型类型: {args.model}")
    if args.model == "mlp":
        print(f"  隐藏层: {args.hidden}")
        print(f"  epochs: {args.epochs}, lr: {args.lr}, patience: {args.patience}")
    print()

    metrics = trainer.run()

    print(f"\n=== 训练完成 ===")
    print(f"  验证准确率: {metrics['best_val_acc']:.4f}")
    print(f"  训练准确率: {metrics['final_train_acc']:.4f}")
    if "epochs_trained" in metrics:
        print(f"  实际训练轮数: {metrics['epochs_trained']}")
    print(f"\n  产出目录: {args.output}")
    print(f"    模型: model.{'pt' if args.model == 'mlp' else 'joblib'}")
    print(f"    元数据: model.meta.json")
    print(f"\n  使用方式:")
    print(f'    engine = TrainedEngine("{args.output}")')
    print(f'    agent = ActionAgent(decision_engine=engine, tool_registry=tools)')


if __name__ == "__main__":
    main()
