"""端到端学习流程测试。

跳过 LLM 标注阶段，用模拟标注数据测试：
  关键帧检测 → 视觉编码 → 标签传播 → MLP 训练 → RL 强化
"""

import json
import time
import random
from pathlib import Path

VIDEO_PATH = "videos/wzry_gameplay_hd.mp4"
OUTPUT_DIR = "runs/e2e_test"

# 模拟的动作空间（王者荣耀）
ACTIONS = [
    "attack", "skill_1", "skill_2", "skill_3",
    "retreat", "move_forward", "last_hit",
    "push_tower", "recall", "idle",
]


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 关键帧检测 ──
    print("=" * 60)
    print("Step 1: 关键帧检测")
    print("=" * 60)

    from vision_agent.core.keyframe import KeyFrameSampler

    sampler = KeyFrameSampler(target_count=300)
    t0 = time.perf_counter()
    keyframes = sampler.sample(VIDEO_PATH)
    kf_time = time.perf_counter() - t0
    print(f"  耗时: {kf_time:.1f}s")
    print(f"  关键帧: {len(keyframes)} 个")

    from collections import Counter
    reasons = Counter(kf.reason for kf in keyframes)
    print(f"  场景切换: {reasons.get('scene_cut', 0)} | "
          f"高变化: {reasons.get('change', 0)} | "
          f"均匀: {reasons.get('uniform', 0)}")

    # ── Step 2: 模拟 LLM 标注 ──
    print()
    print("=" * 60)
    print("Step 2: 模拟 LLM 标注（跳过 API 调用）")
    print("=" * 60)

    # 根据变化分数模拟标注：高变化 → 攻击/技能，低变化 → idle/移动
    annotate_dir = output_dir / "annotations"
    annotate_dir.mkdir(exist_ok=True)
    annotate_path = str(annotate_dir / "annotated_0.jsonl")

    annotated_count = 0
    with open(annotate_path, "w", encoding="utf-8") as f:
        for kf in keyframes:
            # 根据关键帧特征模拟决策
            if kf.score > 0.3:
                # 高变化 → 战斗动作
                action = random.choice(["attack", "skill_1", "skill_2", "skill_3"])
            elif kf.score > 0.1:
                # 中等变化 → 移动/推塔
                action = random.choice(["move_forward", "last_hit", "push_tower"])
            else:
                # 低变化 → 等待/回城
                action = random.choice(["idle", "recall", "retreat"])

            sample = {
                "frame_id": kf.frame_idx,
                "timestamp": round(kf.timestamp, 3),
                "frame_size": [1280, 720],
                "inference_ms": 0,
                "detections": [],
                "object_counts": {},
                "human_action": {
                    "type": "llm_annotated",
                    "action": "press",
                    "key": action,
                    "reason": f"模拟标注 ({kf.reason}, score={kf.score:.3f})",
                },
                "action_timestamp": round(kf.timestamp, 3),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            annotated_count += 1

    print(f"  模拟标注: {annotated_count} 帧")

    # ── Step 3: 端到端训练（编码 + 标签传播 + MLP） ──
    print()
    print("=" * 60)
    print("Step 3: 端到端训练")
    print("=" * 60)

    from vision_agent.workshop.learning_pipeline import LearningPipeline

    pipeline = LearningPipeline(
        output_dir=str(output_dir),
        on_log=lambda msg: print(f"  {msg}"),
        on_progress=lambda phase, pct: None,
    )

    model_dir = str(output_dir / "model")
    t0 = time.perf_counter()
    metrics = pipeline._run_e2e_train(
        video_files=[VIDEO_PATH],
        annotate_dir=str(annotate_dir),
        actions=ACTIONS,
        model_dir=model_dir,
        epochs=50,
    )
    train_time = time.perf_counter() - t0

    print()
    print(f"  总耗时: {train_time:.1f}s")
    print(f"  训练指标: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    # ── Step 4: 推理测试 ──
    print()
    print("=" * 60)
    print("Step 4: 推理测试")
    print("=" * 60)

    if not metrics.get("error"):
        from vision_agent.decision.e2e_engine import E2EEngine
        import cv2

        engine = E2EEngine(model_dir=model_dir)
        engine.on_start()

        # 从视频中取几帧测试
        cap = cv2.VideoCapture(VIDEO_PATH)
        test_frames = []
        for target_sec in [10, 60, 120, 300, 600]:
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
            ret, frame = cap.read()
            if ret:
                test_frames.append((target_sec, frame))
        cap.release()

        for ts, frame in test_frames:
            t0 = time.perf_counter()
            action = engine.decide(None, None, frame=frame)
            latency = (time.perf_counter() - t0) * 1000
            print(f"  @{ts:>4d}s → {action.tool_name:<15s} (conf={action.confidence:.3f}, {latency:.1f}ms)")

        engine.on_stop()

    # ── 总结 ──
    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  关键帧检测: {kf_time:.1f}s")
    print(f"  端到端训练: {train_time:.1f}s")
    if not metrics.get("error"):
        print(f"  最佳验证准确率: {metrics.get('best_val_acc', 0):.1%}")
        print(f"  训练样本数: {metrics.get('train_samples', 0)}")
        print(f"  模型路径: {model_dir}")


if __name__ == "__main__":
    main()
