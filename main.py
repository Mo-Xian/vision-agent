"""Vision Agent 入口。"""

import argparse
import logging
import os
import yaml

from vision_agent.sources import create_source
from vision_agent.core import Detector, Visualizer, ModelManager, StateManager
from vision_agent.core.pipeline import Pipeline
from vision_agent.server import WebSocketServer
from vision_agent.agents import DemoAgent, ActionAgent
from vision_agent.decision import (
    RuleEngine, LLMEngine, TrainedEngine,
    HierarchicalEngine, RLEngine,
    create_provider,
)
from vision_agent.data.recorder import DataRecorder
from vision_agent.tools import ToolRegistry
from vision_agent.tools.keyboard import KeyboardTool
from vision_agent.tools.mouse import MouseTool
from vision_agent.tools.api_call import ApiCallTool
from vision_agent.tools.shell import ShellTool


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def build_tool_registry(config: dict) -> ToolRegistry:
    """根据配置创建工具注册表。"""
    registry = ToolRegistry()
    tools_cfg = config.get("tools", {})
    enabled = tools_cfg.get("enabled", [])

    tool_classes = {
        "keyboard": KeyboardTool,
        "mouse": MouseTool,
        "api_call": ApiCallTool,
    }

    for name in enabled:
        if name == "shell":
            shell_cfg = tools_cfg.get("shell", {})
            allowed = shell_cfg.get("allowed_commands")
            registry.register(ShellTool(allowed_commands=allowed))
        elif name in tool_classes:
            registry.register(tool_classes[name]())

    return registry


def build_decision_engine(config: dict, tool_registry: ToolRegistry):
    """根据配置创建决策引擎。"""
    dec_cfg = config.get("decision", {})
    engine_type = dec_cfg.get("engine", "none")

    if engine_type == "rule":
        rule_cfg = dec_cfg.get("rule", {})
        return RuleEngine(first_match=rule_cfg.get("first_match", True))

    elif engine_type == "trained":
        trained_cfg = dec_cfg.get("trained", {})
        model_dir = trained_cfg.get("model_dir", "runs/decision/exp1")
        confidence = trained_cfg.get("confidence_threshold", 0.3)
        action_key_map = trained_cfg.get("action_key_map", {})
        return TrainedEngine(
            model_dir=model_dir,
            confidence_threshold=confidence,
            action_key_map=action_key_map,
        )

    elif engine_type == "llm":
        llm_cfg = dec_cfg.get("llm", {})
        api_key_env = llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        provider_name = llm_cfg.get("provider", "claude")
        model = llm_cfg.get("model", "claude-sonnet-4-20250514")
        base_url = llm_cfg.get("base_url", "")

        provider = None
        if api_key:
            provider = create_provider(
                provider_name=provider_name,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )

        return LLMEngine(
            provider=provider,
            system_prompt=llm_cfg.get("system_prompt", ""),
            tools_schema=tool_registry.to_claude_tools(),
            decision_interval=llm_cfg.get("decision_interval", 1.0),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )

    elif engine_type == "hierarchical":
        hier_cfg = dec_cfg.get("hierarchical", {})
        micro = RuleEngine(first_match=True)
        return HierarchicalEngine(
            micro=micro,
            strategy_interval=hier_cfg.get("strategy_interval", 5.0),
            tactic_interval=hier_cfg.get("tactic_interval", 1.0),
        )

    elif engine_type == "rl":
        rl_cfg = dec_cfg.get("rl", {})
        actions = rl_cfg.get("actions", ["idle", "attack", "retreat"])
        action_key_map = rl_cfg.get("action_key_map", {})
        return RLEngine(
            actions=actions,
            action_key_map=action_key_map,
            training=rl_cfg.get("training", True),
            save_dir=rl_cfg.get("save_dir", "runs/rl"),
            model_path=rl_cfg.get("model_path", ""),
        )

    return None


def build_model_manager(config: dict) -> ModelManager | None:
    """根据配置创建模型管理器。"""
    models_cfg = config.get("models")
    if not models_cfg:
        return None

    det_cfg = config.get("detector", {})
    mm = ModelManager(detector_kwargs={
        "confidence": det_cfg.get("confidence", 0.5),
        "iou": det_cfg.get("iou", 0.45),
        "classes": det_cfg.get("classes"),
        "device": det_cfg.get("device"),
        "imgsz": det_cfg.get("imgsz", 640),
    })

    for name, path in models_cfg.get("registry", {}).items():
        try:
            mm.register(name, path)
        except FileNotFoundError as e:
            logging.warning(f"跳过不存在的模型: {e}")

    default_name = models_cfg.get("default")
    if default_name and default_name in [m["name"] for m in mm.list_models()]:
        mm.switch(default_name)

    return mm


def main():
    parser = argparse.ArgumentParser(description="Vision Agent - 实时视觉感知与智能决策")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--source", choices=["screen", "camera", "video", "image"], help="覆盖视频源类型")
    parser.add_argument("--model", help="覆盖模型文件 (单模型模式)")
    parser.add_argument("--use-model", help="指定使用的已注册模型名称")
    parser.add_argument("--no-gui", action="store_true", help="禁用可视化窗口")
    parser.add_argument("--no-ws", action="store_true", help="禁用 WebSocket 服务")
    parser.add_argument("--no-agent", action="store_true", help="禁用 Agent")
    parser.add_argument("--agent", choices=["demo", "action"], default="action", help="Agent 类型")
    parser.add_argument("--decision", choices=["rule", "llm", "trained", "hierarchical", "rl", "none"], help="覆盖决策引擎类型")
    parser.add_argument("--decision-model", help="覆盖 trained 引擎的模型目录")
    parser.add_argument("--record", action="store_true", help="启用数据录制模式")
    parser.add_argument("--record-dir", help="覆盖录制数据保存目录")
    parser.add_argument("--session", help="录制会话名称")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    # 命令行覆盖
    if args.source:
        config["source"]["type"] = args.source
    if args.model:
        config["detector"]["model"] = args.model
    if args.decision:
        config["decision"]["engine"] = args.decision
    if args.decision_model:
        config.setdefault("decision", {}).setdefault("trained", {})["model_dir"] = args.decision_model

    # 创建视频源
    source = create_source(config["source"])

    # 创建模型管理器
    model_manager = build_model_manager(config)
    if args.use_model and model_manager:
        model_manager.switch(args.use_model)

    # 创建检测器（ModelManager 未配置或无活跃模型时使用）
    det_cfg = config["detector"]
    detector = Detector(
        model=det_cfg["model"],
        confidence=det_cfg["confidence"],
        iou=det_cfg["iou"],
        classes=det_cfg.get("classes"),
        device=det_cfg.get("device"),
        imgsz=det_cfg.get("imgsz", 640),
    )

    # WebSocket
    ws_server = None
    if not args.no_ws:
        srv_cfg = config.get("server", {})
        ws_server = WebSocketServer(
            host=srv_cfg.get("host", "0.0.0.0"),
            port=srv_cfg.get("port", 8765),
        )

    # 可视化
    visualizer = None
    if not args.no_gui:
        vis_cfg = config.get("visualizer", {})
        if vis_cfg.get("enabled", True):
            visualizer = Visualizer(
                window_name=vis_cfg.get("window_name", "Vision Agent"),
                show_fps=vis_cfg.get("show_fps", True),
                show_labels=vis_cfg.get("show_labels", True),
                show_confidence=vis_cfg.get("show_confidence", True),
            )

    # 状态管理器
    state_manager = StateManager()

    # 数据录制器
    agents = []
    rec_cfg = config.get("recorder", {})
    if args.record or rec_cfg.get("enabled", False):
        recorder = DataRecorder(
            save_dir=args.record_dir or rec_cfg.get("save_dir", "data/recordings"),
            session_name=args.session or rec_cfg.get("session_name"),
            record_mouse_move=rec_cfg.get("record_mouse_move", False),
            max_detection_age=rec_cfg.get("max_detection_age", 0.5),
        )
        agents.append(recorder)
        logging.info(f"数据录制模式已启用 → {recorder._file_path or recorder._save_dir}")

    # Agent
    if not args.no_agent:
        if args.agent == "demo":
            agents.append(DemoAgent())
        else:
            # ActionAgent: 工具 + 决策引擎
            tool_registry = build_tool_registry(config)
            decision_engine = build_decision_engine(config, tool_registry)
            if decision_engine:
                agents.append(ActionAgent(
                    decision_engine=decision_engine,
                    tool_registry=tool_registry,
                    state_manager=state_manager,
                ))
            else:
                agents.append(DemoAgent())

    target_fps = config.get("pipeline", {}).get("target_fps", 0)
    pipeline = Pipeline(
        source=source,
        detector=detector,
        ws_server=ws_server,
        visualizer=visualizer,
        agents=agents,
        state_manager=state_manager,
        model_manager=model_manager,
        target_fps=target_fps,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
