# Vision Agent

## 基本信息

| 属性 | 值 |
|------|-----|
| 项目名 | vision-agent |
| 语言 | Python 3.10+ |
| 核心依赖 | ultralytics (YOLO), OpenCV, PySide6, websockets, pynput, PyTorch |
| 状态 | 活跃开发 |

## 项目定位

实时视觉感知 + 智能决策 + 动作执行框架。核心理念："像人看视频学习"。

## 架构

```
训练工坊(Workshop):
  VideoAnalyzer(视频分析) → AutoAnnotator(LLM标注) → DecisionTrainer(监督训练) → RLEngine(强化学习)
  LearningPipeline(4阶段管线) + ModelRegistry(模型注册表) + LearningSession(会话持久化)

感知层:  Sources → ModelManager → Detector → DetectionResult
         SceneClassifier(场景识别) → ROIExtractor(区域特征)
决策层:  StateManager(状态/空间/ROI) → DecisionEngine → Action
         Rule | Trained(MLP/RF) | LLM | Hierarchical(3层) | RL(DQN)
执行层:  ToolRegistry → Tools(keyboard/mouse/api_call/shell) → ActionAgent
协调层:  AutoPilot(场景→Profile→训练→热加载) + Pipeline + WebSocket + GUI
```

## 核心模块

### Workshop（训练工坊）— 新架构核心

| 文件 | 用途 |
|------|------|
| `vision_agent/workshop/__init__.py` | 模块导出 |
| `vision_agent/workshop/video_analyzer.py` | 视频分析：抽帧→YOLO检测→LLM分析→VideoInsight |
| `vision_agent/workshop/learning_pipeline.py` | 4阶段管线：分析→标注→训练→强化 |
| `vision_agent/workshop/model_registry.py` | 模型注册表：扫描/索引/管理训练模型 |
| `vision_agent/workshop/session.py` | 学习会话持久化 |

### GUI

| 文件 | 用途 |
|------|------|
| `vision_agent/gui/main_window.py` | GUI 主窗口（Workshop/Agent/LLM 三模式） |
| `vision_agent/gui/workshop_panel.py` | 训练工坊面板（视频输入/学习/模型管理） |
| `vision_agent/gui/agent_panel.py` | Agent 执行面板（配置/对话） |
| `vision_agent/gui/llm_panel.py` | LLM 设置面板 |
| `vision_agent/gui/auto_learn_dialog.py` | 自动学习对话框（桥接到 workshop） |

### Core / Decision / Data

| 文件 | 用途 |
|------|------|
| `main.py` | CLI 入口 |
| `gui_app.py` | PySide6 GUI 入口 |
| `config.yaml` | 全局配置 |
| `profiles/*.yaml` | 场景 Profile 配置 |
| `vision_agent/core/detector.py` | YOLO 检测器 |
| `vision_agent/core/pipeline.py` | 主流程管线（支持热切换决策引擎） |
| `vision_agent/core/state.py` | 跨帧状态 + SpatialInfo + EnhancedState |
| `vision_agent/decision/base.py` | Action + DecisionEngine ABC |
| `vision_agent/decision/llm_engine.py` | LLM 决策引擎 |
| `vision_agent/decision/llm_provider.py` | LLM 供应商抽象层（Claude/OpenAI/兼容接口） |
| `vision_agent/decision/trained_engine.py` | 训练模型推理引擎（MLP/RF） |
| `vision_agent/decision/rl_engine.py` | DQN 强化学习引擎 |
| `vision_agent/data/auto_annotator.py` | LLM 自动标注视频帧（Tool Calling） |
| `vision_agent/data/train.py` | ActionMLP + DecisionTrainer |
| `vision_agent/data/recorder.py` | 人工操作录制 |
| `scripts/train_decision.py` | 决策模型训练 CLI |

## 开发命令

```bash
# GUI 模式
python gui_app.py

# CLI 模式
python main.py
python main.py --source screen --model yolov8n.pt

# 录制数据
python main.py --record --record-dir data/recordings

# 训练决策模型
python scripts/train_decision.py --data data/recordings/*.jsonl --output runs/decision/exp1

# 使用训练模型
python main.py --decision trained --decision-model runs/decision/exp1
```

## 数据管线

Workshop 学习管线（推荐）：
1. **视频分析**：`VideoAnalyzer` 抽帧 → YOLO 检测 → LLM 分析 → `VideoInsight`（场景/动作/关键词）
2. **LLM 标注**：`AutoAnnotator` 视频抽帧 → YOLO 检测 → LLM 决策 → JSONL
3. **监督训练**：`DecisionTrainer` JSONL → MLP/RF → `model.pt` + `model.meta.json`
4. **强化学习**：`RLEngine` DQN 在视频帧上训练 → 优化策略

其他方式：
- **人工录制**：`DataRecorder` 捕获键盘/鼠标 + YOLO 检测结果 → JSONL
- **自动学习**：`AutoLearn`（旧）包含资源搜索/下载阶段，推荐用 Workshop 替代

`TrainedEngine` 加载训练模型做实时推理，支持语义动作→实际按键映射（`action_key_map`）。

## 约定

- API key 通过环境变量读取，不写入代码或配置
- 模型文件 `*.pt`、训练数据 `datasets/`、训练输出 `runs/` 通过 `.gitignore` 排除
- 视频文件 `*.mp4` 不纳入版本控制
