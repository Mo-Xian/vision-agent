# Vision Agent

## 基本信息

| 属性 | 值 |
|------|-----|
| 项目名 | vision-agent |
| 语言 | Python 3.10+ |
| 核心依赖 | ultralytics (YOLO), OpenCV, PySide6, websockets, pynput, PyTorch |
| 状态 | 活跃开发 |

## 项目定位

实时视觉感知 + 智能决策 + 动作执行框架。4 层架构：感知层 → 决策层 → 执行层 → 协调层。

## 架构

```
感知层:  Sources(screen/camera/video/image) → ModelManager → Detector → DetectionResult
决策层:  StateManager(跨帧状态) → DecisionEngine(Rule/LLM/Trained) → Action
执行层:  ToolRegistry → Tools(keyboard/mouse/api_call/shell) → ActionAgent
协调层:  Pipeline 串联 + WebSocket + GUI
```

## 核心文件

| 文件 | 用途 |
|------|------|
| `main.py` | CLI 入口 |
| `gui_app.py` | PySide6 GUI 入口 |
| `config.yaml` | 全局配置 |
| `vision_agent/core/detector.py` | YOLO 检测器 |
| `vision_agent/core/pipeline.py` | 主流程管线 |
| `vision_agent/core/model_manager.py` | 模型注册/切换 |
| `vision_agent/core/state.py` | 跨帧场景状态 |
| `vision_agent/core/trainer.py` | YOLO 训练 |
| `vision_agent/decision/base.py` | Action + DecisionEngine ABC |
| `vision_agent/decision/rule_engine.py` | 规则引擎（零延迟） |
| `vision_agent/decision/llm_engine.py` | LLM 决策引擎 |
| `vision_agent/decision/llm_provider.py` | LLM 供应商抽象层（Claude/OpenAI/兼容接口） |
| `vision_agent/decision/trained_engine.py` | 训练模型推理引擎（MLP/RF） |
| `vision_agent/tools/` | 工具：keyboard, mouse, api_call, shell |
| `vision_agent/agents/action_agent.py` | 智能 Agent（决策+工具） |
| `vision_agent/data/recorder.py` | 人工操作录制（键盘/鼠标 + 检测结果） |
| `vision_agent/data/auto_annotator.py` | LLM 自动标注视频帧 |
| `vision_agent/data/dataset.py` | 数据集加载与特征提取 |
| `vision_agent/data/train.py` | ActionMLP + DecisionTrainer |
| `vision_agent/gui/main_window.py` | GUI 主窗口 |
| `vision_agent/gui/annotate_dialog.py` | LLM 自动标注对话框 |
| `scripts/train_decision.py` | 决策模型训练 CLI |

## 开发命令

```bash
# GUI 模式
python gui_app.py

# CLI 模式
python main.py
python main.py --source screen --model yolov8n.pt
python main.py --no-gui

# 录制数据
python main.py --record --record-dir data/recordings

# 训练决策模型
python scripts/train_decision.py --data data/recordings/*.jsonl --output runs/decision/exp1

# 使用训练模型
python main.py --decision trained --decision-model runs/decision/exp1
```

## 数据管线

两种训练数据获取方式：
1. **人工录制**：`DataRecorder` 捕获键盘/鼠标 + YOLO 检测结果 → JSONL
2. **LLM 自动标注**：`AutoAnnotator` 视频抽帧 → YOLO 检测 → LLM 决策 → JSONL

训练产出：`model.pt` + `model.meta.json`

`TrainedEngine` 加载训练模型做实时推理，支持语义动作→实际按键映射（`action_key_map`）。

## 约定

- API key 通过环境变量读取，不写入代码或配置
- 模型文件 `*.pt`、训练数据 `datasets/`、训练输出 `runs/` 通过 `.gitignore` 排除
- 视频文件 `*.mp4` 不纳入版本控制
