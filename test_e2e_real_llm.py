"""端到端学习流程测试 —— 使用真实 MiniMax VLM 标注。

完整流程:
  Phase 1: LLM 视觉分析视频 (MCP VLM 理解截图 → M2.7 分析场景)
  Phase 2: LLM 标注关键帧 (MCP VLM 理解截图 → M2.7 决策标注)
  Phase 3: 视觉编码 + 标签传播 + MLP 训练
  Phase 4: 策略梯度 RL 强化
"""

import os
import time
from pathlib import Path

VIDEO_PATH = "videos/wzry_gameplay_hd.mp4"
OUTPUT_DIR = "runs/e2e_real_llm"

# MiniMax 配置 - 从环境变量读取
PROVIDER = "minimax"
API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MODEL = "MiniMax-M2.7"

# 加载王者荣耀知识
KNOWLEDGE_PATH = "knowledge/wzry_5v5.txt"


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 加载知识
    knowledge = ""
    kp = Path(KNOWLEDGE_PATH)
    if kp.exists():
        knowledge = kp.read_text(encoding="utf-8")
        print(f"已加载场景知识: {len(knowledge)} 字符")
    else:
        print(f"知识文件不存在: {KNOWLEDGE_PATH}")

    t_start = time.perf_counter()

    from vision_agent.workshop.learning_pipeline import LearningPipeline

    pipeline = LearningPipeline(
        llm_provider_name=PROVIDER,
        llm_api_key=API_KEY,
        llm_model=MODEL,
        output_dir=OUTPUT_DIR,
        on_log=lambda msg: print(f"  {msg}"),
        on_progress=lambda phase, pct: print(f"  [{phase}] {pct:.0%}") if int(pct * 100) % 20 == 0 else None,
    )

    result = pipeline.learn_from_videos(
        video_paths=[VIDEO_PATH],
        description="王者荣耀5v5",
        sample_count=50,          # 50 个关键帧标注
        analyze_samples=10,       # 分析阶段抽 10 帧
        batch_size=5,             # 每次发 5 帧给 LLM
        epochs=80,                # MLP 训练轮数
        rl_steps=1000,            # RL 强化步数
        send_image=True,
        knowledge=knowledge,
        e2e=True,                 # 端到端模式
    )

    total_time = time.perf_counter() - t_start

    # ── 结果汇总 ──
    print()
    print("=" * 60)
    print("完整学习流程结果")
    print("=" * 60)
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  成功: {result.success}")
    print(f"  运行目录: {result.run_dir}")

    if result.insight:
        print(f"  场景识别: {result.insight.scene_type}")
        print(f"  动作空间: {result.insight.suggested_actions}")

    print(f"  LLM 标注: {result.annotated_count} 帧")

    if result.metrics:
        print(f"  训练样本: {result.metrics.get('train_samples', '?')}")
        print(f"  验证准确率: {result.metrics.get('best_val_acc', 0):.1%}")

    if result.phases.get("rl"):
        rl = result.phases["rl"]
        print(f"  RL 步数: {rl.get('total_steps', 0)}")
        print(f"  RL 平均奖励: {rl.get('avg_reward', 0):.4f}")

    print(f"  模型目录: {result.model_dir}")
    print(f"  Profile: {result.profile_path}")

    # ── 推理测试 ──
    if result.success and result.model_dir:
        print()
        print("=" * 60)
        print("推理测试")
        print("=" * 60)

        import cv2
        from vision_agent.decision.e2e_engine import E2EEngine

        engine = E2EEngine(model_dir=result.model_dir)
        engine.on_start()

        cap = cv2.VideoCapture(VIDEO_PATH)
        for target_sec in [10, 30, 60, 120, 300, 600]:
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            t0 = time.perf_counter()
            action = engine.decide(None, None, frame=frame)
            latency = (time.perf_counter() - t0) * 1000
            print(f"  @{target_sec:>4d}s -> {action.tool_name:<20s} conf={action.confidence:.3f}  {latency:.1f}ms")
        cap.release()
        engine.on_stop()


if __name__ == "__main__":
    main()
