"""测试脚本：用王者荣耀视频测试检测 + 规则决策管线。
不控制键盘/鼠标，只输出检测结果和决策动作。
"""

import time
import json
import logging
from vision_agent.sources import create_source
from vision_agent.core.detector import Detector, DetectionResult
from vision_agent.core.state import StateManager
from vision_agent.decision.base import Action, DecisionEngine
from vision_agent.decision.rule_engine import RuleEngine
from vision_agent.tools.base import BaseTool, ToolResult, ToolRegistry
from vision_agent.agents.action_agent import ActionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")


class DryRunTool(BaseTool):
    """空操作工具：只记录动作，不实际执行。"""

    def __init__(self, tool_name: str):
        self._name = tool_name
        self.history: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"DryRun {self._name}"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs) -> ToolResult:
        record = {"tool": self._name, "params": kwargs, "time": time.strftime("%H:%M:%S")}
        self.history.append(record)
        logger.info(f"  [DryRun] {self._name}: {json.dumps(kwargs, ensure_ascii=False)}")
        return ToolResult(success=True)


def build_rule_engine() -> RuleEngine:
    """构建简单的规则引擎：检测到目标就攻击，否则idle。"""
    engine = RuleEngine()

    def rule_attack(result: DetectionResult, state):
        if not result.detections:
            return None
        # 有目标时攻击
        det = result.detections[0]
        return Action(
            tool_name="keyboard",
            parameters={"action": "press", "key": "a"},
            reason=f"发现 {det.class_name} (conf={det.confidence:.2f})",
        )

    engine.add_rule("attack_on_detect", rule_attack)
    return engine


def main():
    video_path = "test_wzry.mp4"

    logger.info(f"=== Vision Agent 管线测试 ===")
    logger.info(f"视频: {video_path}")
    logger.info(f"模式: DryRun（不执行真实操作）")
    logger.info("")

    # 1. 创建视频源
    source = create_source({"type": "video", "video": {"path": video_path, "loop": False}})

    # 2. 创建检测器（用默认 yolov8n）
    detector = Detector(model="yolov8n.pt", confidence=0.3)

    # 3. DryRun 工具注册
    registry = ToolRegistry()
    kb_tool = DryRunTool("keyboard")
    ms_tool = DryRunTool("mouse")
    registry.register(kb_tool)
    registry.register(ms_tool)

    # 4. 规则引擎
    engine = build_rule_engine()

    # 5. StateManager + ActionAgent
    state_mgr = StateManager()
    agent = ActionAgent(
        decision_engine=engine,
        tool_registry=registry,
        state_manager=state_mgr,
        on_log=lambda msg: logger.info(f"  {msg}"),
    )

    # 6. 运行管线
    source.start()
    agent.on_start()

    frame_count = 0
    max_frames = 150  # 测试 150 帧
    stats = {"total_frames": 0, "frames_with_detections": 0, "total_detections": 0}

    logger.info("--- 开始检测 ---")
    t0 = time.time()

    try:
        while frame_count < max_frames:
            frame = source.read()
            if frame is None:
                logger.info("视频结束")
                break

            result = detector.detect(frame)
            frame_count += 1
            stats["total_frames"] = frame_count

            if result.detections:
                stats["frames_with_detections"] += 1
                stats["total_detections"] += len(result.detections)

                if frame_count % 10 == 1:  # 每 10 帧输出一次详情
                    names = [f"{d.class_name}({d.confidence:.2f})" for d in result.detections[:5]]
                    logger.info(f"[Frame {frame_count:>4}] {len(result.detections)} 目标: {', '.join(names)}  |  推理: {result.inference_ms:.1f}ms")

            # 发送给 Agent
            agent.on_detection(result)
            time.sleep(0.01)  # 模拟帧间隔

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        agent.on_stop()
        source.stop()

    elapsed = time.time() - t0
    fps = frame_count / max(elapsed, 0.001)

    logger.info("")
    logger.info("=== 测试结果 ===")
    logger.info(f"处理帧数: {stats['total_frames']}")
    logger.info(f"有目标帧: {stats['frames_with_detections']} ({100*stats['frames_with_detections']/max(stats['total_frames'],1):.1f}%)")
    logger.info(f"总检测数: {stats['total_detections']}")
    logger.info(f"耗时: {elapsed:.1f}s ({fps:.1f} FPS)")
    logger.info(f"DryRun 动作执行次数: {len(kb_tool.history)}")
    if kb_tool.history:
        logger.info(f"最近 5 条动作:")
        for record in kb_tool.history[-5:]:
            logger.info(f"  [{record['time']}] {record['tool']}: {record['params']}")


if __name__ == "__main__":
    main()
