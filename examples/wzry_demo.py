"""王者荣耀 Demo：检测→决策→执行 全链路演示。

演示如何：
1. 加载王者荣耀训练模型
2. 定义游戏规则（看到什么→做什么）
3. ActionAgent 自动执行决策

使用方式：
    python examples/wzry_demo.py                    # 分析视频（不执行动作）
    python examples/wzry_demo.py --live              # 实时屏幕捕获 + 执行动作
    python examples/wzry_demo.py --source video      # 分析视频文件
"""

import sys
import os
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_agent.sources import create_source
from vision_agent.core import Detector, ModelManager, StateManager
from vision_agent.core.pipeline import Pipeline
from vision_agent.decision import RuleEngine, Action
from vision_agent.tools import ToolRegistry
from vision_agent.tools.base import BaseTool, ToolResult
from vision_agent.tools.keyboard import KeyboardTool
from vision_agent.tools.mouse import MouseTool
from vision_agent.agents import ActionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wzry_demo")


# ============================================================
# 自定义工具：日志动作（安全的，不执行真实操作）
# ============================================================

class LogActionTool(BaseTool):
    """将决策动作记录到日志（不执行真实操作，适合测试）。"""

    @property
    def name(self):
        return "log_action"

    @property
    def description(self):
        return "记录一个计划执行的动作到日志（不实际执行）"

    @property
    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "description": "动作类型"},
                "detail": {"type": "string", "description": "动作详情"},
            },
            "required": ["action_type", "detail"],
        }

    def execute(self, action_type: str, detail: str, **kwargs) -> ToolResult:
        logger.info(f"  >> [动作] {action_type}: {detail}")
        return ToolResult(success=True, output={"action_type": action_type, "detail": detail})


# ============================================================
# 游戏规则定义
# ============================================================

def rule_low_health_retreat(result, state):
    """规则：检测到己方低血量(1格血)时，建议撤退。"""
    if "blood_ally_1q" in state.object_counts:
        return Action(
            tool_name="log_action",
            parameters={"action_type": "撤退", "detail": "己方英雄低血量(1格)，建议撤退"},
            priority=10,
        )
    return None


def rule_enemy_nearby(result, state):
    """规则：检测到敌方小兵较多时，提示注意。"""
    enemy_count = state.object_counts.get("enemy_minion", 0)
    if enemy_count >= 3:
        return Action(
            tool_name="log_action",
            parameters={"action_type": "警戒", "detail": f"前方发现 {enemy_count} 个敌方小兵"},
            priority=5,
        )
    return None


def rule_skill_ready(result, state):
    """规则：检测到技能就绪时，提示可以释放。"""
    standby = state.object_counts.get("skill_standby", 0)
    cooling = state.object_counts.get("skill_cooling", 0)
    if standby >= 2 and cooling == 0:
        return Action(
            tool_name="log_action",
            parameters={"action_type": "进攻", "detail": f"{standby} 个技能就绪，可以发起进攻"},
            priority=3,
        )
    return None


def rule_heal_available(result, state):
    """规则：治疗技能就绪 + 低血量时使用。"""
    has_heal = "heal_standby" in state.object_counts
    low_hp = ("blood_ally_1q" in state.object_counts or "blood_ally_2q" in state.object_counts)
    if has_heal and low_hp:
        return Action(
            tool_name="log_action",
            parameters={"action_type": "治疗", "detail": "血量较低且治疗就绪，建议使用治疗"},
            priority=8,
        )
    return None


def rule_tower_danger(result, state):
    """规则：检测到敌方防御塔时警告。"""
    if "enemy_tower" in state.object_counts:
        return Action(
            tool_name="log_action",
            parameters={"action_type": "警告", "detail": "前方有敌方防御塔，注意走位"},
            priority=7,
        )
    return None


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="王者荣耀 Demo - 视觉检测+决策")
    parser.add_argument("--live", action="store_true", help="实时屏幕捕获模式")
    parser.add_argument("--source", default="video", choices=["video", "screen", "image"])
    parser.add_argument("--video", default="test_wzry.mp4", help="视频文件路径")
    parser.add_argument("--execute", action="store_true", help="执行真实键鼠操作（谨慎！）")
    args = parser.parse_args()

    # 1. 视频源
    if args.live:
        source_cfg = {"type": "screen", "screen": {"monitor": 1, "fps": 10}}
    elif args.source == "image":
        source_cfg = {"type": "image", "image": {"path": "datasets/honor_of_kings/val/images"}}
    else:
        source_cfg = {"type": "video", "video": {"path": args.video, "loop": False}}

    source = create_source(source_cfg)

    # 2. 模型（优先用自定义模型，否则用通用模型）
    model_path = "yolov8n.pt"
    custom_model = os.environ.get("YOLO_MODEL", "")
    if custom_model and os.path.exists(custom_model):
        model_path = custom_model

    detector = Detector(model=model_path, confidence=0.5, imgsz=640)
    logger.info(f"模型: {model_path}")

    # 3. 工具
    registry = ToolRegistry()
    registry.register(LogActionTool())
    if args.execute:
        logger.warning("⚠ 已启用真实键鼠操作！")
        registry.register(KeyboardTool())
        registry.register(MouseTool())

    # 4. 规则引擎
    engine = RuleEngine(first_match=False)  # 收集所有匹配规则
    engine.add_rule("low_health_retreat", rule_low_health_retreat)
    engine.add_rule("heal_available", rule_heal_available)
    engine.add_rule("tower_danger", rule_tower_danger)
    engine.add_rule("enemy_nearby", rule_enemy_nearby)
    engine.add_rule("skill_ready", rule_skill_ready)
    logger.info(f"已加载 5 条游戏规则")

    # 5. Agent
    state_mgr = StateManager(history_size=10)
    agent = ActionAgent(
        decision_engine=engine,
        tool_registry=registry,
        state_manager=state_mgr,
    )

    # 6. 启动 Pipeline
    logger.info("=" * 50)
    logger.info("王者荣耀 Demo 启动")
    logger.info(f"输入源: {source_cfg['type']}")
    logger.info(f"工具: {registry.tool_names}")
    logger.info("=" * 50)

    pipeline = Pipeline(
        source=source, detector=detector,
        agents=[agent], state_manager=state_mgr,
        visualizer=None, ws_server=None,
    )
    pipeline.run()

    logger.info("=" * 50)
    logger.info(f"Agent 统计: {agent.stats}")
    logger.info("Demo 结束")


if __name__ == "__main__":
    main()
