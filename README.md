# Vision Agent

实时视觉感知 + 智能决策 + 动作执行框架。

## 架构

```
感知层:  Sources(screen/camera/video/image) → ModelManager → Detector → DetectionResult
         SceneClassifier(场景识别) → ROIExtractor(区域特征)
决策层:  StateManager(跨帧状态/空间/ROI) → DecisionEngine → Action
         ├─ Rule     (规则引擎，零延迟)
         ├─ Trained  (MLP/RF，毫秒级)
         ├─ LLM      (Claude/OpenAI/本地，秒级)
         ├─ Hierarchical (分层：战略→战术→操作)
         └─ RL       (DQN 强化学习，自主探索)
执行层:  ToolRegistry → Tools(keyboard/mouse/api_call/shell) → ActionAgent
协调层:  AutoPilot(场景识别→Profile路由→自动训练→热加载)
         Pipeline 串联 + WebSocket + PySide6 GUI
```

## 特性

- **多视频源**：屏幕捕获、摄像头、视频文件、图片目录
- **实时检测**：基于 YOLOv8，支持自定义训练模型和模型切换
- **多决策引擎**：规则引擎（零延迟）、LLM 引擎（Claude/OpenAI/Qwen/DeepSeek/Ollama）、训练模型引擎（MLP/RF）、分层引擎（战略/战术/操作）、DQN 强化学习引擎
- **场景 Profile 系统**：YAML 配置场景（动作列表、按键映射、ROI 区域、自动训练参数），内置王者荣耀/FPS/桌面模板
- **AutoPilot 自动闭环**：场景识别 → Profile 路由 → LLM 自动标注 → 模型训练 → 热加载决策引擎，全程无人工干预
- **动作执行**：键盘/鼠标模拟、API 调用、Shell 命令
- **数据管线**：人工录制 → 训练 / LLM 自动标注 → 训练 → 实时推理
- **Tool Calling 规范化输出**：通过函数调用 + enum 约束强制 LLM 返回结构化结果
- **增强状态管理**：空间关系计算（质心/面积/距离）、ROI 区域特征（血条/蓝条/小地图）、场景分类（时序平滑）
- **API 安全**：API Key 仅从环境变量读取，不做持久化存储
- **WebSocket 推送**：结构化 JSON 结果实时推送
- **PySide6 GUI**：深色科技感主题，支持配置、预览、录制、训练、标注、场景管理

## 快速开始

### 方式一：EXE 直接运行（推荐）

从 [Releases](../../releases) 下载最新版压缩包，解压后双击 `VisionAgent.exe` 即可，无需安装 Python。

### 方式二：源码运行

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

### 方式三：自行打包 EXE

```bash
# 一键打包（自动创建 venv + 安装依赖 + PyInstaller 打包）
build.bat

# 或手动打包
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
python build_exe.py

# 产出在 dist/VisionAgent/ 目录下
dist\VisionAgent\VisionAgent.exe
```

打包信息：
- 使用 CPU 版 PyTorch，体积约 **886 MB**
- 运行内存约 **400 MB**
- 不需要目标机器安装 Python
- 将 `dist/VisionAgent/` 整个目录打包分发即可

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

#### LLM 标注输出规范化

标注器支持两种 LLM 输出模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Tool Calling**（默认） | 通过函数调用 + `enum` 约束，LLM 被强制从预设动作列表中选择，返回结构化 JSON | Claude、GPT-4o、Qwen 等支持 function calling 的模型 |
| **文本解析**（回退） | 从 LLM 自由文本中提取 JSON 或关键词匹配，4 层 fallback 机制 | Ollama 本地模型等不支持 tool calling 的场景 |

Tool Calling 模式下，LLM 收到的工具定义：
```json
{
  "name": "decide_action",
  "input_schema": {
    "properties": {
      "action": { "type": "string", "enum": ["attack", "retreat", "skill_1", "idle"] },
      "reason": { "type": "string" }
    },
    "required": ["action"]
  }
}
```

LLM 返回的结构化结果（由 API 层面保证格式）：
```json
{
  "name": "decide_action",
  "input": { "action": "attack", "reason": "检测到敌方英雄在近处" }
}
```

其他标注选项：
- **发送帧图像**：勾选后将视频帧（JPEG）一并发送给多模态 LLM，利用视觉信息辅助决策
- **速率限制与重试**：自动控制请求间隔，遇到 429/5xx 错误指数退避重试
- **训练数据保护**：不在预设列表中的动作自动丢弃，避免污染训练集

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
├── build.bat                        # 一键打包 EXE
├── build_exe.py                     # PyInstaller 打包配置
├── config.yaml                      # 全局配置
├── profiles/                        # 场景 Profile 配置
│   ├── wzry_5v5.yaml                # 王者荣耀 5v5
│   ├── fps_generic.yaml             # 通用 FPS 射击
│   └── desktop.yaml                 # 桌面通用
├── scripts/
│   └── train_decision.py            # 决策模型训练脚本
├── examples/
│   ├── run_demo.py                  # 快速启动示例
│   └── wzry_demo.py                 # 游戏规则引擎示例
├── vision_agent/
│   ├── core/
│   │   ├── detector.py              # YOLO 检测器
│   │   ├── pipeline.py              # 主流程管线（支持热切换）
│   │   ├── model_manager.py         # 模型注册/切换
│   │   ├── state.py                 # 跨帧状态 + 空间信息 + 增强状态
│   │   ├── scene_classifier.py      # 场景自动分类（时序平滑）
│   │   ├── roi_extractor.py         # ROI 区域特征提取
│   │   ├── trainer.py               # YOLO 训练
│   │   └── visualizer.py            # OpenCV 可视化
│   ├── decision/
│   │   ├── base.py                  # Action + DecisionEngine ABC
│   │   ├── rule_engine.py           # 规则引擎（零延迟）
│   │   ├── llm_engine.py            # LLM 决策引擎
│   │   ├── llm_provider.py          # LLM 供应商抽象层
│   │   ├── trained_engine.py        # 训练模型引擎（MLP/RF）
│   │   ├── hierarchical.py          # 分层决策（战略→战术→操作）
│   │   └── rl_engine.py             # DQN 强化学习引擎
│   ├── profiles/
│   │   ├── base.py                  # SceneProfile + ProfileManager
│   │   └── loader.py                # YAML 配置加载/保存
│   ├── auto/
│   │   ├── auto_trainer.py          # 自动训练管线（标注→训练）
│   │   └── auto_pilot.py            # 自动驾驶编排器
│   ├── data/
│   │   ├── recorder.py              # 人工操作录制
│   │   ├── auto_annotator.py        # LLM 自动标注（Tool Calling）
│   │   ├── dataset.py               # 数据集加载
│   │   └── train.py                 # MLP/RF 训练
│   ├── tools/                       # 键盘/鼠标/API/Shell
│   ├── agents/                      # 智能 Agent
│   ├── sources/                     # 视频源 (screen/camera/video/image)
│   ├── server/                      # WebSocket 服务
│   └── gui/                         # PySide6 GUI 组件
├── tests/
│   └── test_auto_annotator.py       # 自动标注单元测试
└── requirements.txt
```

## 场景 Profile

Profile 是预定义的场景配置，存放在 `profiles/` 目录下：

```yaml
name: wzry_5v5
display_name: 王者荣耀 5v5
yolo_model: runs/detect/wzry/weights/best.pt
actions: [attack, retreat, skill_1, skill_2, skill_3, ultimate, recall, idle]
action_key_map:
  attack: {type: key, key: a}
  skill_1: {type: key, key: "1"}
roi_regions:
  hp_bar: [0.42, 0.92, 0.58, 0.95]
  minimap: [0.0, 0.7, 0.2, 1.0]
scene_keywords: [hero, tower, minion, monster]
auto_train:
  enabled: true
  sample_count: 500
  llm_provider: claude
```

**AutoPilot 闭环流程**：检测到目标 → 场景分类器识别场景 → 匹配 Profile → 自动标注帧数据 → 训练决策模型 → 热加载到 Pipeline → 实时决策

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
