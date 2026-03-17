# Vision Agent

实时视觉感知 + 智能决策 + 动作执行框架。

## 架构

```
感知层:  Sources(screen/camera/video/image) → ModelManager → Detector → DetectionResult
决策层:  StateManager(跨帧状态) → DecisionEngine(Rule/LLM/Trained) → Action
执行层:  ToolRegistry → Tools(keyboard/mouse/api_call/shell) → ActionAgent
协调层:  Pipeline 串联 + WebSocket + PySide6 GUI
```

## 特性

- **多视频源**：屏幕捕获、摄像头、视频文件、图片目录
- **实时检测**：基于 YOLOv8，支持自定义训练模型和模型切换
- **多决策引擎**：规则引擎（零延迟）、LLM 引擎（Claude/OpenAI/Qwen/DeepSeek/Ollama）、训练模型引擎（MLP/RandomForest）
- **动作执行**：键盘/鼠标模拟、API 调用、Shell 命令
- **数据管线**：人工录制 → 训练 / LLM 自动标注 → 训练 → 实时推理
- **WebSocket 推送**：结构化 JSON 结果实时推送
- **PySide6 GUI**：完整图形界面，支持配置、预览、录制、训练、标注

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# GUI 模式
python gui_app.py

# CLI 模式
python main.py
python main.py --source screen --model yolov8n.pt
python main.py --no-gui

# Windows 快速启动（自动创建 venv）
start.bat
```

## 数据管线

### 方式一：人工录制训练
```bash
# 录制人工操作（键盘/鼠标 + YOLO 检测结果）
python main.py --record --record-dir data/recordings

# 训练决策模型
python scripts/train_decision.py --data data/recordings/*.jsonl --output runs/decision/exp1

# 使用训练模型
python main.py --decision trained --decision-model runs/decision/exp1
```

### 方式二：LLM 自动标注
1. GUI 中打开「录制与训练」标签页
2. 点击「LLM 自动标注」，配置视频/模型/LLM
3. 标注完成后点击「开始训练」
4. 切换决策引擎到 `trained`，配置动作映射

## 配置

编辑 `config.yaml` 自定义：
- 视频源类型和参数
- YOLO 模型和检测参数
- 决策引擎（rule/llm/trained）
- 动作映射（语义名→实际按键）
- WebSocket 服务端口

## 项目结构

```
vision-agent/
├── main.py                          # CLI 入口
├── gui_app.py                       # PySide6 GUI 入口
├── config.yaml                      # 全局配置
├── detect_video.py                  # 独立视频检测脚本
├── scripts/
│   └── train_decision.py            # 决策模型训练脚本
├── tools/
│   └── capture.py                   # 截图采集工具
├── examples/
│   ├── run_demo.py                  # 快速启动示例
│   └── wzry_demo.py                 # 游戏规则引擎示例
├── vision_agent/
│   ├── core/
│   │   ├── detector.py              # YOLO 检测器
│   │   ├── pipeline.py              # 主流程管线
│   │   ├── model_manager.py         # 模型注册/切换
│   │   ├── state.py                 # 跨帧场景状态
│   │   ├── trainer.py               # YOLO 训练
│   │   └── visualizer.py            # OpenCV 可视化
│   ├── decision/
│   │   ├── base.py                  # Action + DecisionEngine ABC
│   │   ├── rule_engine.py           # 规则引擎
│   │   ├── llm_engine.py            # LLM 决策引擎
│   │   ├── llm_provider.py          # LLM 供应商抽象层
│   │   └── trained_engine.py        # 训练模型引擎
│   ├── data/
│   │   ├── recorder.py              # 人工操作录制
│   │   ├── auto_annotator.py        # LLM 自动标注
│   │   ├── dataset.py               # 数据集加载
│   │   └── train.py                 # MLP/RF 训练
│   ├── tools/
│   │   ├── keyboard.py              # 键盘模拟
│   │   ├── mouse.py                 # 鼠标模拟
│   │   ├── api_call.py              # HTTP API 调用
│   │   └── shell.py                 # Shell 命令
│   ├── agents/
│   │   ├── action_agent.py          # 智能 Agent
│   │   └── demo_agent.py            # 示例 Agent
│   ├── sources/                     # 视频源 (screen/camera/video/image)
│   ├── server/                      # WebSocket 服务
│   └── gui/                         # PySide6 GUI 组件
│       ├── main_window.py           # 主窗口
│       ├── annotate_dialog.py       # LLM 标注对话框
│       ├── train_dialog.py          # YOLO 训练对话框
│       └── video_widget.py          # 视频预览组件
└── requirements.txt
```

## WebSocket 数据格式

连接 `ws://localhost:8765` 接收实时检测数据：

```json
{
  "frame_id": 42,
  "timestamp": 1710000000.123,
  "inference_ms": 12.5,
  "frame_size": [1920, 1080],
  "count": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [100.0, 200.0, 300.0, 500.0],
      "bbox_norm": [0.0521, 0.1852, 0.1563, 0.4630]
    }
  ]
}
```

## License

MIT
