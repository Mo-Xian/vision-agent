"""Vision Agent CLI — 行为克隆学习框架。

用法:
    # PC 游戏录制（F9 暂停/恢复）
    python main.py record --output recordings/session1 --fps 10
    python main.py record --window "王者荣耀"  # 窗口捕获

    # 手机游戏录制（scrcpy + ADB）
    python main.py mobile --game moba                    # MOBA 默认区域
    python main.py mobile --zones touch_zones.json       # 自定义区域
    python main.py mobile --check                        # 检查环境

    # 远程 PC 录制（服务端-客户端架构）
    python main.py hub                                   # 本机启动中转服务，等待客户端连接并录制
    python main.py hub --port 9876 -o recordings/remote  # 指定端口和输出目录
    # 远程 PC 运行客户端: RemoteCaptureClient.exe ws://本机IP:9876

    # 从录制数据学习
    python main.py learn-bc recordings/session1 recordings/session2

    # 伪标签扩展：用已有模型标注新视频，扩充数据
    python main.py expand runs/workshop/exp1/model video1.mp4 video2.mp4

    # RL 自对弈学习
    python main.py self-play --preset wzry                          # 王者荣耀内置预设
    python main.py self-play --preset wzry --bc-model runs/.../model  # BC 热启动
    python main.py self-play --game moba --zones zones.json         # 自定义

    # 评估模型
    python main.py eval runs/workshop/exp1/model --video video.mp4
"""

import argparse
import logging
import os
import sys


def cmd_record(args):
    """录制游戏操作。"""
    from vision_agent.data.game_recorder import GameRecorder

    if args.list_windows:
        windows = GameRecorder.list_windows()
        print(f"可见窗口 ({len(windows)} 个):")
        for w in windows:
            print(f"  {w}")
        return

    recorder = GameRecorder(
        output_dir=args.output,
        fps=args.fps,
        window_title=args.window,
        record_mouse=not args.no_mouse,
        hotkey=args.hotkey,
        on_log=lambda msg: print(msg),
    )

    window_info = f", 窗口: {args.window}" if args.window else ", 全屏"
    mouse_info = " + 鼠标" if not args.no_mouse else ""
    print(f"开始录制... (FPS={args.fps}{window_info}, 键盘{mouse_info})")
    print(f"按 {args.hotkey.upper()} 暂停/恢复，Ctrl+C 停止")
    recorder.start()

    try:
        import time
        while recorder.is_recording:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    stats = recorder.stop()
    print(f"\n录制完成!")
    print(f"  帧数: {stats.total_frames}")
    print(f"  时长: {stats.duration_sec:.0f}s")
    print(f"  事件: {stats.total_events}")
    print(f"  输出: {stats.output_dir}")


def cmd_learn_bc(args):
    """从录制数据学习（行为克隆）。"""
    from vision_agent.workshop.learning_pipeline import LearningPipeline

    api_key = args.api_key or os.environ.get("MINIMAX_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key and args.provider != "ollama":
        print("提示: 未配置 LLM，动作发现将直接使用按键名")

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

    result = pipeline.learn_from_recordings(
        recording_dirs=args.recordings,
        description=args.description,
        epochs=args.epochs,
        rl_steps=args.rl_steps,
        knowledge=knowledge,
    )

    if result.success:
        print(f"\n学习成功!")
        print(f"  模型: {result.model_dir}")
        print(f"  Profile: {result.profile_path}")
        print(f"  验证精度: {result.metrics.get('best_val_acc', 0):.3f}")
        if result.coach_advice.get("suggestions"):
            print(f"\n教练建议:")
            for s in result.coach_advice["suggestions"][:3]:
                print(f"  - {s}")
    else:
        print("\n学习失败")
        sys.exit(1)


def cmd_expand(args):
    """伪标签扩展：用已有模型标注新视频。"""
    from vision_agent.workshop.learning_pipeline import LearningPipeline

    pipeline = LearningPipeline(
        output_dir=args.output,
        on_log=lambda msg: print(msg),
    )

    result = pipeline.expand_from_videos(
        model_dir=args.model_dir,
        video_paths=args.videos,
        confidence_threshold=args.threshold,
        max_idle_ratio=args.max_idle,
        epochs=args.epochs,
        mix_ratio=args.mix_ratio,
    )

    if result.success:
        metrics = result.metrics
        print(f"\n扩展成功!")
        print(f"  新模型: {result.model_dir}")
        print(f"  验证精度: {metrics.get('best_val_acc', 0):.3f}")
        print(f"  总样本: {metrics.get('total_samples', 0)}"
              f" (原始 {metrics.get('orig_samples', 0)}"
              f" + 伪标签 {metrics.get('pseudo_samples', 0)})")
    else:
        print("\n扩展失败")
        sys.exit(1)


def cmd_mobile(args):
    """手机游戏录制（scrcpy + ADB）。"""
    from vision_agent.data.mobile_recorder import MobileRecorder

    if args.check:
        ok, msgs = MobileRecorder.check_prerequisites()
        for m in msgs:
            print(m)
        if ok:
            print("\n环境检查通过，可以开始录制!")
        else:
            print("\n环境检查未通过，请根据上方提示修复。")
            sys.exit(1)
        return

    # 加载触控区域
    zones = None
    if args.zones:
        import json
        with open(args.zones, "r", encoding="utf-8") as f:
            zones = json.load(f)
        print(f"已加载触控区域配置: {args.zones}")
    elif args.game:
        zones = MobileRecorder.create_default_zones(args.game)
        print(f"使用 {args.game} 默认触控区域")

    recorder = MobileRecorder(
        output_dir=args.output,
        fps=args.fps,
        zones=zones,
        device_serial=args.device,
        on_log=lambda msg: print(msg),
    )

    print(f"开始手机录制... (FPS={args.fps})")
    print("按 Ctrl+C 停止")
    recorder.start()

    try:
        import time
        while recorder.is_recording:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    stats = recorder.stop()
    print(f"\n录制完成!")
    print(f"  帧数: {stats.total_frames}")
    print(f"  时长: {stats.duration_sec:.0f}s")
    print(f"  触控事件: {stats.total_events}")
    print(f"  输出: {stats.output_dir}")


def cmd_hub(args):
    """启动中转服务并录制远程客户端画面。"""
    from vision_agent.data.remote_hub import RemoteHub
    from vision_agent.data.remote_recorder import RemoteRecorder
    import time

    relay_url = getattr(args, 'relay', '') or ''
    room_id = getattr(args, 'room', '') or ''
    token = getattr(args, 'token', '') or ''

    hub = RemoteHub(
        port=args.port,
        relay_url=relay_url,
        room_id=room_id,
        relay_token=token,
        on_log=lambda msg: print(msg),
    )
    hub.start()

    if hub.is_relay_mode:
        print(f"\n已连接公网中继: {relay_url}")
        print(f"房间: {hub.room_id}")
        print(f"远程客户端连接: {relay_url}  房间={hub.room_id}")
    else:
        local_ip = hub.get_local_ip()
        print(f"\n中转服务已启动: ws://{local_ip}:{args.port}")
        print(f"请在远程设备连接此地址")
    print(f"等待远程客户端连接...")
    print(f"按 Ctrl+C 停止\n")

    # 等待客户端连接
    try:
        while not hub.is_client_connected:
            time.sleep(0.5)
    except KeyboardInterrupt:
        hub.stop()
        print("\n已停止")
        return

    print(f"客户端已连接: {hub.client_addr}")
    print(f"开始录制...\n")

    recorder = RemoteRecorder(
        hub=hub,
        output_dir=args.output,
        on_log=lambda msg: print(msg),
    )
    recorder.start()

    try:
        while recorder.is_recording:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    stats = recorder.stop()
    hub.stop()
    print(f"\n录制完成!")
    print(f"  帧数: {stats.total_frames}")
    print(f"  时长: {stats.duration_sec:.0f}s")
    print(f"  事件: {stats.total_events}")
    print(f"  输出: {stats.output_dir}")


def cmd_self_play(args):
    """RL 自对弈学习。"""
    import json as _json
    from vision_agent.rl.self_play import SelfPlayLoop
    from vision_agent.rl.preset import load_selfplay_preset

    # 加载预设或自定义配置
    if args.preset:
        preset = load_selfplay_preset(args.preset)
        action_zones = preset["action_zones"]
        reward_config = preset["reward_config"]
        dqn_params = preset["dqn_params"]
        bc_model = args.bc_model or preset.get("bc_model_dir", "")
        output = args.output if args.output != "runs/selfplay/exp1" else preset.get("output_dir", args.output)
        print(f"使用预设: {args.preset}")
    elif args.zones:
        with open(args.zones, "r", encoding="utf-8") as f:
            raw_zones = _json.load(f)
        action_zones = [{"name": "idle"}]
        for name, zone in raw_zones.items():
            action_zones.append({"name": name, **zone})
        from vision_agent.rl.reward import RewardConfig
        reward_config = RewardConfig()
        dqn_params = {}
        bc_model = args.bc_model
        output = args.output
    elif args.game:
        from vision_agent.data.mobile_recorder import MobileRecorder
        raw_zones = MobileRecorder.create_default_zones(args.game)
        action_zones = [{"name": "idle"}]
        for name, zone in raw_zones.items():
            action_zones.append({"name": name, **zone})
        from vision_agent.rl.reward import RewardConfig
        reward_config = RewardConfig()
        dqn_params = {}
        bc_model = args.bc_model
        output = args.output
    else:
        print("错误: 需要指定 --preset、--game 或 --zones")
        sys.exit(1)

    print(f"动作空间: {[z['name'] for z in action_zones]}")

    start_model = preset.get("start_model_path", "models/start.onnx") if args.preset else "models/start.onnx"

    loop = SelfPlayLoop(
        action_zones=action_zones,
        bc_model_dir=bc_model,
        output_dir=output,
        device_serial=args.device,
        reward_config=reward_config,
        start_model_path=start_model,
        lr=dqn_params.get("lr", args.lr),
        gamma=dqn_params.get("gamma", args.gamma),
        epsilon_start=dqn_params.get("epsilon_start", args.epsilon),
        buffer_capacity=dqn_params.get("buffer_capacity", args.buffer_size),
        batch_size=dqn_params.get("batch_size", args.batch_size),
        fps=args.fps,
        save_interval=args.save_interval,
        max_episodes=args.max_episodes,
        on_log=lambda msg: print(msg),
    )

    loop.start()

    try:
        import time
        while loop.is_running:
            time.sleep(5)
            s = loop.stats
            print(
                f"  [统计] 对局={s['episodes']} 步数={s['total_steps']} "
                f"训练={s['train_steps']} ε={s['epsilon']:.3f} "
                f"近10局奖励={s['avg_reward_10ep']:.1f}"
            )
    except KeyboardInterrupt:
        pass

    loop.stop()
    print(f"\n自对弈结束! 模型保存至: {args.output}")


def cmd_eval(args):
    """评估训练模型。"""
    from eval_model import main as eval_main
    sys.argv = ["eval_model", "--model", args.model_dir, args.mode]
    if args.video:
        sys.argv.extend(["--video", args.video])
    eval_main()


def main():
    parser = argparse.ArgumentParser(description="Vision Agent — 端到端视频学习框架")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # record 子命令
    p_rec = sub.add_parser("record", help="录制游戏操作")
    p_rec.add_argument("-o", "--output", default="recordings/session", help="输出目录")
    p_rec.add_argument("--fps", type=int, default=10, help="截屏帧率 (默认: 10)")
    p_rec.add_argument("-w", "--window", default="", help="游戏窗口标题（模糊匹配），空=全屏")
    p_rec.add_argument("--no-mouse", action="store_true", help="不录制鼠标操作")
    p_rec.add_argument("--hotkey", default="f9", help="暂停/恢复快捷键 (默认: F9)")
    p_rec.add_argument("--list-windows", action="store_true", help="列出所有可见窗口标题")

    # mobile 子命令（手机录制）
    p_mob = sub.add_parser("mobile", help="手机游戏录制（scrcpy + ADB）")
    p_mob.add_argument("-o", "--output", default="recordings/mobile", help="输出目录")
    p_mob.add_argument("--fps", type=int, default=10, help="截屏帧率 (默认: 10)")
    p_mob.add_argument("-g", "--game", default="", help="游戏预设 (moba/fps)")
    p_mob.add_argument("-z", "--zones", default="", help="触控区域 JSON 文件路径")
    p_mob.add_argument("-d", "--device", default="", help="ADB 设备序列号（多设备时指定）")
    p_mob.add_argument("--check", action="store_true", help="检查环境（adb/scrcpy/设备）")

    # hub 子命令（启动中转服务 + 录制远程客户端）
    p_hub = sub.add_parser("hub", help="启动中转服务，等待远程客户端连接并录制")
    p_hub.add_argument("-p", "--port", type=int, default=9876, help="中转服务端口 (默认: 9876)")
    p_hub.add_argument("-o", "--output", default="recordings/remote", help="输出目录")
    p_hub.add_argument("--relay", default="", help="公网中继地址 (如 ws://my-server:9877)")
    p_hub.add_argument("--room", default="", help="中继房间 ID (留空自动生成)")
    p_hub.add_argument("--token", default="", help="中继 token (可选)")

    # learn-bc 子命令（行为克隆）
    p_bc = sub.add_parser("learn-bc", help="从录制数据学习（行为克隆，推荐）")
    p_bc.add_argument("recordings", nargs="+", help="录制目录路径")
    p_bc.add_argument("-d", "--description", default="", help="场景描述")
    p_bc.add_argument("-p", "--provider", default="minimax", help="LLM 供应商 (教练用)")
    p_bc.add_argument("-m", "--model", default="MiniMax-M2.7", help="LLM 模型")
    p_bc.add_argument("--api-key", default="", help="API Key")
    p_bc.add_argument("--base-url", default="", help="自定义 Base URL")
    p_bc.add_argument("-o", "--output", default="runs/workshop", help="输出目录")
    p_bc.add_argument("-e", "--epochs", type=int, default=100, help="训练轮数")
    p_bc.add_argument("--rl-steps", type=int, default=0, help="RL 步数 (0=跳过)")
    p_bc.add_argument("-k", "--knowledge", default="", help="场景知识文件路径")

    # expand 子命令（伪标签扩展）
    p_exp = sub.add_parser("expand", help="伪标签扩展（用已有模型标注新视频）")
    p_exp.add_argument("model_dir", help="已训练模型目录")
    p_exp.add_argument("videos", nargs="+", help="新视频文件路径")
    p_exp.add_argument("-t", "--threshold", type=float, default=0.85, help="置信度阈值 (默认: 0.85)")
    p_exp.add_argument("--max-idle", type=float, default=0.3, help="idle 最大比例 (默认: 0.3)")
    p_exp.add_argument("-e", "--epochs", type=int, default=100, help="重新训练轮数")
    p_exp.add_argument("--mix-ratio", type=float, default=0.3, help="原始数据混入比例 (默认: 0.3)")
    p_exp.add_argument("-o", "--output", default="runs/workshop", help="输出目录")

    # self-play 子命令（RL 自对弈）
    p_sp = sub.add_parser("self-play", help="RL 自对弈学习（可从 BC 模型热启动）")
    p_sp.add_argument("-p", "--preset", default="", help="游戏预设 (wzry/moba) 或 YAML 文件路径")
    p_sp.add_argument("-g", "--game", default="", help="游戏类型 (moba/fps)")
    p_sp.add_argument("-z", "--zones", default="", help="触控区域 JSON 文件路径")
    p_sp.add_argument("--bc-model", default="", help="BC 预训练模型目录（热启动）")
    p_sp.add_argument("-o", "--output", default="runs/selfplay/exp1", help="输出目录")
    p_sp.add_argument("-d", "--device", default="", help="ADB 设备序列号")
    p_sp.add_argument("--fps", type=int, default=5, help="决策帧率 (默认: 5)")
    p_sp.add_argument("--lr", type=float, default=0.0005, help="学习率")
    p_sp.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p_sp.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    p_sp.add_argument("--buffer-size", type=int, default=50000, help="经验缓冲区大小")
    p_sp.add_argument("--batch-size", type=int, default=64, help="训练批大小")
    p_sp.add_argument("--save-interval", type=int, default=50, help="保存间隔（训练步数）")
    p_sp.add_argument("--max-episodes", type=int, default=0, help="最大对局数 (0=无限)")

    # eval 子命令
    p_eval = sub.add_parser("eval", help="评估训练模型")
    p_eval.add_argument("model_dir", help="模型目录")
    p_eval.add_argument("--video", default="", help="视频文件")
    p_eval.add_argument("--mode", default="stats", choices=["video", "curve", "stats"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "record":
        cmd_record(args)
    elif args.command == "mobile":
        cmd_mobile(args)
    elif args.command == "hub":
        cmd_hub(args)
    elif args.command == "learn-bc":
        cmd_learn_bc(args)
    elif args.command == "expand":
        cmd_expand(args)
    elif args.command == "self-play":
        cmd_self_play(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
