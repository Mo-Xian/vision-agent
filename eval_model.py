"""模型效果评估工具。

3 种评估方式:
  1. 视频推理预览 — 在视频上逐帧推理，输出带决策标注的视频
  2. 训练曲线 — 可视化 loss / accuracy 变化
  3. 密集推理统计 — 在视频上密集推理，统计动作分布和置信度

用法:
  python eval_model.py                         # 全部评估
  python eval_model.py video                   # 只生成标注视频
  python eval_model.py curve                   # 只画训练曲线
  python eval_model.py stats                   # 只跑密集统计
  python eval_model.py --model runs/xxx/model  # 指定模型目录
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 默认路径
DEFAULT_MODEL_DIR = "runs/e2e_real_llm/20260326_131852/model"
DEFAULT_VIDEO = "videos/wzry_gameplay_hd.mp4"
DEFAULT_OUTPUT_DIR = "runs/eval_output"


def load_engine(model_dir: str):
    """加载 E2E 推理引擎。"""
    from vision_agent.decision.e2e_engine import E2EEngine
    engine = E2EEngine(model_dir=model_dir)
    engine.on_start()
    return engine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 生成带决策标注的视频
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 动作 → 颜色映射
ACTION_COLORS = {
    "attack":  (0, 0, 255),     # 红
    "defend":  (255, 165, 0),   # 橙
    "idle":    (200, 200, 200), # 灰
    "skill_1": (0, 255, 255),   # 黄
    "skill_2": (255, 0, 255),   # 品红
    "skill_3": (255, 0, 0),     # 蓝
    "retreat": (0, 255, 0),     # 绿
}

def get_color(action: str) -> tuple:
    return ACTION_COLORS.get(action, (255, 255, 255))


def eval_video(model_dir: str, video_path: str, output_dir: str,
               sample_interval: int = 5, max_seconds: int = 120):
    """在视频上推理并生成带标注的输出视频。"""
    print("=" * 55)
    print("  评估 1: 生成带决策标注的视频")
    print("=" * 55)

    engine = load_engine(model_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * fps)

    out_path = str(Path(output_dir) / "eval_annotated.mp4")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps / sample_interval, (w, h))

    print(f"  视频: {Path(video_path).name} ({w}x{h}, {fps:.0f}fps)")
    print(f"  采样: 每 {sample_interval} 帧推理 1 次")
    print(f"  最大: {max_seconds}s ({max_frames} 帧)")
    print(f"  输出: {out_path}")
    print()

    frame_idx = 0
    infer_count = 0
    t_start = time.perf_counter()
    last_action = ""
    last_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        frame_idx += 1
        if frame_idx % sample_interval != 0:
            continue

        # 推理
        actions = engine.decide(embedding=frame)
        action = actions[0] if actions else None
        last_action = action.name if action else "unknown"
        last_conf = action.confidence if action else 0.0
        infer_count += 1

        # 绘制标注
        annotated = frame.copy()
        color = get_color(last_action)
        ts = frame_idx / fps

        # 顶部半透明背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # 动作文字
        label = f"{last_action} ({last_conf:.1%})"
        cv2.putText(annotated, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # 时间戳
        time_str = f"{int(ts//60):02d}:{ts%60:05.2f}"
        cv2.putText(annotated, time_str, (w - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 置信度条
        bar_w = int(300 * last_conf)
        cv2.rectangle(annotated, (20, 62), (20 + bar_w, 75), color, -1)
        cv2.rectangle(annotated, (20, 62), (320, 75), (100, 100, 100), 1)

        writer.write(annotated)

        if infer_count % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  ... {infer_count} 帧推理完成 ({elapsed:.1f}s, @{ts:.1f}s)")

    writer.release()
    cap.release()
    engine.on_stop()

    elapsed = time.perf_counter() - t_start
    print(f"  完成: {infer_count} 帧, {elapsed:.1f}s")
    print(f"  平均: {elapsed/max(infer_count,1)*1000:.1f}ms/帧")
    print(f"  输出视频: {out_path}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 训练曲线可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eval_curve(model_dir: str, output_dir: str):
    """用 OpenCV 绘制训练曲线图。"""
    print("=" * 55)
    print("  评估 2: 训练曲线")
    print("=" * 55)

    history_path = Path(model_dir) / "train_history.json"
    if not history_path.exists():
        print(f"  训练历史不存在: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    epochs = len(train_loss)

    if epochs == 0:
        print("  无训练数据")
        return

    meta_path = Path(model_dir) / "model.meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"  训练轮数: {epochs}")
    print(f"  最佳验证准确率: {meta.get('best_val_acc', max(val_acc) if val_acc else 0):.1%}")
    print(f"  动作数: {meta.get('num_actions', '?')}")
    print(f"  训练样本: {meta.get('train_samples', '?')}")

    # 绘制图表（纯 OpenCV，不依赖 matplotlib）
    W, H = 800, 500
    margin = 60
    pw = W - margin * 2  # 绘图区宽
    ph = H - margin * 2  # 绘图区高

    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 标题
    cv2.putText(img, "Training Curves", (W // 2 - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 坐标轴
    ox, oy = margin, H - margin  # 原点
    cv2.line(img, (ox, margin), (ox, oy), (0, 0, 0), 2)
    cv2.line(img, (ox, oy), (W - margin, oy), (0, 0, 0), 2)

    # X 轴标签
    for i in range(0, epochs + 1, max(1, epochs // 5)):
        x = ox + int(i / max(epochs - 1, 1) * pw)
        cv2.line(img, (x, oy), (x, oy + 5), (0, 0, 0), 1)
        cv2.putText(img, str(i), (x - 8, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "Epoch", (W // 2 - 20, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Y 轴标签 (0 ~ 1)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = oy - int(v * ph)
        cv2.line(img, (ox - 5, y), (ox, y), (0, 0, 0), 1)
        cv2.putText(img, f"{v:.2f}", (5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        if v > 0:
            cv2.line(img, (ox, y), (W - margin, y), (220, 220, 220), 1)

    def draw_curve(data, color, max_val=1.0):
        pts = []
        for i, v in enumerate(data):
            x = ox + int(i / max(len(data) - 1, 1) * pw)
            y = oy - int(min(v / max_val, 1.0) * ph)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, 2)

    # 绘制曲线
    max_loss = max(train_loss) if train_loss else 1.0
    draw_curve(train_loss, (0, 0, 200), max_val=max_loss)      # 红: loss
    draw_curve(train_acc, (200, 100, 0), max_val=1.0)           # 蓝: train_acc
    draw_curve(val_acc, (0, 180, 0), max_val=1.0)               # 绿: val_acc

    # 图例
    ly = margin + 10
    for label, color in [("Loss (scaled)", (0, 0, 200)),
                         ("Train Acc", (200, 100, 0)),
                         ("Val Acc", (0, 180, 0))]:
        cv2.line(img, (W - 180, ly), (W - 150, ly), color, 2)
        cv2.putText(img, label, (W - 145, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        ly += 20

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "train_curves.png")
    cv2.imwrite(out_path, img)
    print(f"  训练曲线图: {out_path}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 密集推理统计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eval_stats(model_dir: str, video_path: str, output_dir: str,
               sample_interval: int = 30, max_seconds: int = 600):
    """在视频上密集推理，输出动作分布和时间线。"""
    print("=" * 55)
    print("  评估 3: 密集推理统计")
    print("=" * 55)

    engine = load_engine(model_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps)

    action_counts = {}
    confidence_sum = {}
    timeline = []  # [(timestamp, action, confidence)]

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        frame_idx += 1
        if frame_idx % sample_interval != 0:
            continue

        actions = engine.decide(embedding=frame)
        action = actions[0] if actions else None
        if action is None:
            continue
        name = action.name
        conf = action.confidence
        ts = frame_idx / fps

        action_counts[name] = action_counts.get(name, 0) + 1
        confidence_sum[name] = confidence_sum.get(name, 0.0) + conf
        timeline.append({"time": round(ts, 1), "action": name, "confidence": round(conf, 3)})

    cap.release()
    engine.on_stop()

    elapsed = time.perf_counter() - t_start
    total_infer = sum(action_counts.values())

    print(f"  推理帧数: {total_infer}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/max(total_infer,1)*1000:.1f}ms/帧)")
    print()

    # 动作分布
    print("  动作分布:")
    print(f"  {'动作':<20s} {'次数':>6s} {'占比':>8s} {'平均置信度':>10s}")
    print(f"  {'-'*48}")
    for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / max(total_infer, 1)
        avg_conf = confidence_sum[name] / count
        bar = "#" * int(pct * 30)
        print(f"  {name:<20s} {count:>6d} {pct:>7.1%} {avg_conf:>10.3f}  {bar}")

    # 时间线摘要（每 30 秒统计一次主要动作）
    print()
    print("  时间线摘要（每30s主导动作）:")
    window = 30
    for start in range(0, int(max_seconds), window):
        end = start + window
        seg = [t for t in timeline if start <= t["time"] < end]
        if not seg:
            continue
        seg_counts = {}
        for t in seg:
            seg_counts[t["action"]] = seg_counts.get(t["action"], 0) + 1
        dominant = max(seg_counts, key=seg_counts.get)
        dom_pct = seg_counts[dominant] / len(seg)
        print(f"  {start:>4d}-{end:>4d}s: {dominant:<15s} ({dom_pct:.0%}, {len(seg)} 帧)")

    # 保存详细结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = {
        "model_dir": model_dir,
        "video": video_path,
        "total_infer": total_infer,
        "action_distribution": {
            name: {
                "count": count,
                "ratio": round(count / max(total_infer, 1), 4),
                "avg_confidence": round(confidence_sum[name] / count, 4),
            }
            for name, count in action_counts.items()
        },
        "timeline": timeline,
    }
    out_path = str(Path(output_dir) / "eval_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  详细数据: {out_path}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    import argparse
    parser = argparse.ArgumentParser(description="E2E 模型效果评估")
    parser.add_argument("modes", nargs="*", default=["curve", "stats", "video"],
                        help="评估模式: video / curve / stats")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="模型目录")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="测试视频")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--max-seconds", type=int, default=120, help="视频最大秒数")
    args = parser.parse_args()

    print(f"模型: {args.model}")
    print(f"视频: {args.video}")
    print()

    if "curve" in args.modes:
        eval_curve(args.model, args.output)

    if "stats" in args.modes:
        eval_stats(args.model, args.video, args.output, max_seconds=args.max_seconds)

    if "video" in args.modes:
        eval_video(args.model, args.video, args.output, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
