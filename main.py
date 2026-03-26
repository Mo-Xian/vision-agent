"""Vision Agent CLI — 端到端视频学习。

用法:
    python main.py learn video1.mp4 video2.mp4 --provider minimax --model MiniMax-M2.7
    python main.py eval runs/workshop/20260326_131852/model --video video.mp4
"""

import argparse
import logging
import os
import sys


def cmd_learn(args):
    """从视频学习决策模型。"""
    from vision_agent.workshop.learning_pipeline import LearningPipeline

    api_key = args.api_key or os.environ.get("MINIMAX_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and args.provider != "ollama":
        print("错误: 请通过 --api-key 或环境变量提供 API Key")
        sys.exit(1)

    knowledge = ""
    if args.knowledge:
        with open(args.knowledge, "r", encoding="utf-8") as f:
            knowledge = f.read()

    pipeline = LearningPipeline(
        llm_provider_name=args.provider,
        llm_api_key=api_key,
        llm_model=args.model,
        llm_base_url=args.base_url,
        output_dir=args.output,
        on_log=lambda msg: print(msg),
    )

    result = pipeline.learn_from_videos(
        video_paths=args.videos,
        description=args.description,
        sample_count=args.samples,
        epochs=args.epochs,
        rl_steps=args.rl_steps,
        send_image=not args.no_image,
        batch_size=args.batch_size,
        knowledge=knowledge,
    )

    if result.success:
        print(f"\n学习成功!")
        print(f"  模型: {result.model_dir}")
        print(f"  Profile: {result.profile_path}")
        print(f"  验证精度: {result.metrics.get('best_val_acc', 0):.3f}")
    else:
        print("\n学习失败")
        sys.exit(1)


def cmd_eval(args):
    """评估训练模型。"""
    # 转发到 eval_model.py
    from eval_model import main as eval_main
    sys.argv = ["eval_model", "--model-dir", args.model_dir, "--mode", args.mode]
    if args.video:
        sys.argv.extend(["--video", args.video])
    eval_main()


def main():
    parser = argparse.ArgumentParser(description="Vision Agent — 端到端视频学习框架")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # learn 子命令
    p_learn = sub.add_parser("learn", help="从视频学习决策模型")
    p_learn.add_argument("videos", nargs="+", help="视频文件路径")
    p_learn.add_argument("-d", "--description", default="", help="场景描述")
    p_learn.add_argument("-p", "--provider", default="minimax", help="LLM 供应商 (默认: minimax)")
    p_learn.add_argument("-m", "--model", default="MiniMax-M2.7", help="LLM 模型")
    p_learn.add_argument("--api-key", default="", help="API Key (也可用环境变量)")
    p_learn.add_argument("--base-url", default="", help="自定义 Base URL")
    p_learn.add_argument("-o", "--output", default="runs/workshop", help="输出目录")
    p_learn.add_argument("-s", "--samples", type=int, default=300, help="标注帧数")
    p_learn.add_argument("-e", "--epochs", type=int, default=100, help="训练轮数")
    p_learn.add_argument("--rl-steps", type=int, default=2000, help="RL 步数 (0=跳过)")
    p_learn.add_argument("--batch-size", type=int, default=5, help="LLM 批量标注大小")
    p_learn.add_argument("--no-image", action="store_true", help="不发送截图给 LLM")
    p_learn.add_argument("-k", "--knowledge", default="", help="场景知识文件路径")

    # eval 子命令
    p_eval = sub.add_parser("eval", help="评估训练模型")
    p_eval.add_argument("model_dir", help="模型目录")
    p_eval.add_argument("--video", default="", help="视频文件（生成标注视频）")
    p_eval.add_argument("--mode", default="stats", choices=["video", "curve", "stats"], help="评估模式")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "learn":
        cmd_learn(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
