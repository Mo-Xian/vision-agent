# Vision Agent

一个基于计算机视觉的智能自动化框架。通过 YOLO 实时识别画面中的目标，结合 AI 决策引擎自动做出反应并执行操作（按键、鼠标、API 调用等）。

**核心思路**：看到什么 → 判断该做什么 → 自动去做。整个过程可以从零开始自动学习，不需要手写规则。

## 能做什么

### 游戏 AI
- 识别游戏画面中的角色、敌人、技能等目标
- 自动决策攻击、释放技能、撤退等操作
- 内置王者荣耀、FPS 射击游戏场景模板，一键配置

### 桌面自动化
- 识别屏幕上的 UI 元素（按钮、输入框、图标等）
- 自动点击、输入文字、执行快捷键
- 适合重复性桌面操作的自动化

### 视频/图像分析
- 对视频流或图片进行实时目标检测
- 统计目标数量、跟踪位置变化
- 通过 WebSocket 将检测结果推送给其他系统

### 自动学习闭环
- **不需要手写规则**：录一段视频 → LLM 自动标注「这个画面应该做什么」→ 训练出决策模型 → 实时运行
- **AutoPilot 模式**：自动识别当前场景 → 匹配配置 → 自动收集数据 → 自动训练 → 自动部署，全程无需人工干预

## 功能列表

| 功能 | 说明 |
|------|------|
| **YOLO 实时检测** | 支持 YOLOv8 全系列模型（n/s/m/l），可自定义训练 |
| **多输入源** | 屏幕捕获、摄像头、视频文件、图片目录、RTSP 流 |
| **规则引擎** | 基于 if-else 规则的决策，零延迟响应 |
| **LLM 决策** | 接入 Claude / OpenAI / Qwen / DeepSeek / Ollama，用大模型判断该做什么 |
| **训练模型决策** | 用 MLP 或 RandomForest 训练轻量决策模型，毫秒级响应 |
| **分层决策** | 战略层（5s 一次）→ 战术层（1s 一次）→ 操作层（每帧），各层独立配置引擎 |
| **强化学习** | DQN 引擎，自主探索 + 经验回放，从环境反馈中学习 |
| **场景 Profile** | YAML 配置文件定义场景（动作、按键、ROI 区域），快速切换不同场景 |
| **AutoPilot** | 自动识别场景 → 匹配 Profile → LLM 标注 → 训练模型 → 热加载，全自动 |
| **人工录制** | 一边操作一边录制键盘/鼠标 + 检测结果，生成训练数据 |
| **LLM 自动标注** | 视频抽帧 → YOLO 检测 → 发给 LLM 判断动作 → 生成训练数据（支持 Tool Calling） |
| **YOLO 训练** | GUI 内直接配置数据集和参数，一键训练自定义检测模型 |
| **动作执行** | 键盘模拟、鼠标模拟、HTTP API 调用、Shell 命令 |
| **ROI 提取** | 从固定区域提取特征（血条比例、颜色、亮度），辅助决策 |
| **WebSocket** | 实时推送检测结果 JSON，供外部系统对接 |
| **GUI 界面** | 深色科技感主题，配置/预览/录制/训练/标注/场景管理一体化 |
| **EXE 打包** | 一键打包为独立可执行文件，无需 Python 环境 |

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

## 使用场景示例

### 场景一：游戏自动操作

1. 打开 GUI → 输入源选「屏幕捕获」
2. 场景 Tab 选择「王者荣耀 5v5」Profile
3. 检测模型选择训练好的模型（或用预训练的 yolov8n.pt 先体验）
4. 决策引擎选 `rule`（规则）或 `trained`（训练模型）
5. 点击「启动检测」，程序自动识别画面并操作

### 场景二：从零开始训练

1. 准备一段游戏视频
2. GUI →「录制/训练」Tab → 点击「LLM 自动标注」
3. 配置视频路径、YOLO 模型、LLM（如 Claude）
4. LLM 会逐帧分析画面并标注「这一帧应该做什么」
5. 标注完成后点击「开始训练」，生成决策模型
6. 切换决策引擎到 `trained`，配置动作→按键映射
7. 启动检测，模型自动决策

### 场景三：全自动 AutoPilot

1. 在 `profiles/` 下创建场景 YAML 配置
2. GUI → 场景 Tab → 勾选「启用 AutoPilot」
3. 启动检测，系统自动：
   - 识别当前画面属于哪个场景
   - 加载对应的 Profile 配置
   - 收集帧数据 → LLM 标注 → 训练模型
   - 训练完成后自动切换到训练好的决策引擎

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
| **Tool Calling**（默认） | 通过函数调用 + `enum` 约束，LLM 被强制从预设动作列表中选择 | Claude、GPT-4o、Qwen 等支持 function calling 的模型 |
| **文本解析**（回退） | 从 LLM 自由文本中提取 JSON 或关键词匹配 | Ollama 本地模型等不支持 tool calling 的场景 |

Tool Calling 模式下 LLM 收到的工具定义：
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

内置 3 个模板：
- `wzry_5v5.yaml` — 王者荣耀 5v5（8 动作，含血条/小地图/技能栏 ROI）
- `fps_generic.yaml` — 通用 FPS 射击（12 动作，含准心/血量/弹药 ROI）
- `desktop.yaml` — 桌面通用（8 动作，使用预训练 yolov8n.pt）

## 配置

编辑 `config.yaml` 自定义：
- 视频源类型和参数
- YOLO 模型和检测参数
- 决策引擎（rule/llm/trained/hierarchical/rl）
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
│   ├── core/                        # 核心：检测、状态、场景分类、ROI
│   ├── decision/                    # 决策引擎：Rule/LLM/Trained/Hierarchical/RL
│   ├── profiles/                    # 场景 Profile 管理
│   ├── auto/                        # AutoPilot + AutoTrainer
│   ├── data/                        # 数据录制、LLM 标注、训练
│   ├── tools/                       # 键盘/鼠标/API/Shell
│   ├── agents/                      # 智能 Agent
│   ├── sources/                     # 视频源 (screen/camera/video/image)
│   ├── server/                      # WebSocket 服务
│   └── gui/                         # PySide6 GUI 组件
├── tests/
│   └── test_auto_annotator.py       # 单元测试
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

## 技术栈

| 组件 | 技术 |
|------|------|
| 目标检测 | YOLOv8 (ultralytics) |
| 图像处理 | OpenCV |
| 深度学习 | PyTorch |
| 机器学习 | scikit-learn |
| GUI | PySide6 (Qt) |
| LLM 接入 | Claude API / OpenAI API / 兼容接口 |
| 输入模拟 | pynput |
| 屏幕捕获 | mss |
| 实时通信 | WebSocket |
| 打包分发 | PyInstaller |

## License

MIT
