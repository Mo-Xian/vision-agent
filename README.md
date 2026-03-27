# Vision Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-41cd52.svg)](https://doc.qt.io/qtforpython/)

**端到端行为克隆 + RL 自学习框架，用于游戏 AI 和桌面自动化。**

录制人类操作 → 行为克隆训练 → 伪标签扩展 → RL 自对弈，全流程 GUI 可视化。支持 PC（窗口捕获 + 键鼠）和手机（scrcpy + ADB 触控）。

> **核心思路**：看人怎么玩 → 模仿学会 → 自己练习变强。从零开始，不需要手写规则。

---

## 界面预览

### 训练工坊

录制操作、行为克隆训练、伪标签扩展、RL 自对弈，全部在一个面板完成。

![训练工坊](docs/screenshots/mode_train.png)

### Agent 部署

连接手机设备，加载训练好的模型（BC 或 DQN），Agent 自动接管游戏操作并显示实时画面。

![Agent 部署](docs/screenshots/mode_agent.png)

### LLM 配置

LLM 连接配置，训练工坊的 LLM Coach 和伪标签扩展共用此配置。

![LLM 配置](docs/screenshots/mode_llm.png)

---

## 架构：统一学习管线

像人一样学习：**老师教 → 自己学 → 实战练**。一键启动，自动编排全流程。

```
Phase 1 教师教学:
  人类录制操作 → 自动检测为人类操作 → LLM 动作发现 → BC 行为克隆训练

Phase 2 自主学习:
  ├─ 技能差距分析 — LLM 教练评估还缺什么
  ├─ 在线搜索 — 搜索相关教学视频资源
  └─ 智能伪标签 — 对新视频逐帧分类:
       模型自信(>85%) → 直接标注
       中等置信度 → 询问 LLM 教练标注
       LLM 也不确定 → 标记待人工补充

Phase 3 自我实践:
  BC 热启动 → 手机自对弈 (scrcpy + ADB)
  → DQN ε-greedy 探索 → 奖励检测(血条/击杀/胜负)
  → 经验回放 → Q 网络训练 → 持续改进

部署:
  训练产出 → Agent 接管 → ADB 触控(手机) 或 pynput(PC)
```

三个阶段的模型格式统一（`model.pt` + `model.meta.json`），可互相衔接：
- 教师教学的产出可用于自主学习的伪标签和自我实践的热启动
- RL 产出可反过来用于伪标签扩展
- 模型浏览器自动扫描所有产出，一目了然

---

## 决策引擎

| 引擎 | 延迟 | 说明 |
|------|------|------|
| **E2EEngine** | 毫秒级 | 行为克隆模型推理，MobileNetV3 编码 + MLP 分类 |
| **DQNEngine** | 毫秒级 | DQN 强化学习模型推理，支持 ADB 触控和 PC 键鼠 |
| **LLM Coach** | 秒级 | Claude / OpenAI / DeepSeek / MiniMax 等，用于动作发现和诊断 |

---

## 快速开始

### 源码运行

```bash
# 安装依赖
pip install -r requirements.txt

# GPU 支持（可选）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# GUI 模式（推荐）
python gui_app.py

# Windows 快速启动（自动创建 venv）
start.bat
```

### CLI 模式

```bash
# PC 录制（F9 暂停/恢复，Ctrl+C 停止）
python main.py record --output recordings/session1 --fps 10
python main.py record --window "王者荣耀"

# 手机录制（scrcpy + ADB）
python main.py mobile --check                        # 检查环境
python main.py mobile --game moba                    # MOBA 预设

# 行为克隆训练
python main.py learn-bc recordings/session1 recordings/session2

# 伪标签扩展
python main.py expand runs/workshop/exp1/model video1.mp4 video2.mp4

# RL 自对弈
python main.py self-play --preset wzry                            # 王者荣耀预设
python main.py self-play --preset wzry --bc-model runs/.../model  # BC 热启动

# 评估模型
python main.py eval runs/workshop/exp1/model --mode stats
```

### EXE 打包

```bash
build.bat
# 产出在 dist/VisionAgent/
```

---

## 界面说明

界面分为 3 个模式，通过顶部按钮切换：

| 模式 | 用途 | 核心操作 |
|------|------|----------|
| **训练工坊** | 数据录制、BC 训练、伪标签扩展、RL 自对弈 | 选窗口录制、一键学习、启动自对弈 |
| **Agent 部署** | 连接手机设备、加载模型、Agent 接管 | 刷新设备、选择模型、Agent 接管/停止 |
| **LLM 设置** | LLM 连接配置 | 选择供应商/模型、填写 API Key、测试连接 |

---

## 场景预设

预设配置存放在 `profiles/` 目录下：

| 预设 | 文件 | 用途 |
|------|------|------|
| 王者荣耀 5v5 | `wzry_5v5.yaml` | BC 训练用，动作列表 + 按键映射 + ROI 区域 |
| 王者荣耀自对弈 | `wzry_selfplay.yaml` | RL 自对弈用，触控区域 + 奖励检测区域 + DQN 参数 |
| FPS 射击 | `fps_generic.yaml` | FPS 游戏 BC 训练 |
| FPS 自对弈 | `fps_selfplay.yaml` | FPS 游戏 RL 自对弈 |
| 桌面通用 | `desktop.yaml` | 通用桌面自动化 |

### 预设配置示例

```yaml
name: wzry_5v5
display_name: 王者荣耀 5v5
actions: [attack, retreat, skill_1, skill_2, skill_3, ultimate, recall, idle]
action_key_map:
  attack: {type: key, key: a}
  skill_1: {type: key, key: "1"}
touch_zones:
  attack: {x: 0.85, y: 0.72, w: 0.08, h: 0.08}
  joystick: {x: 0.15, y: 0.72, w: 0.15, h: 0.15}
reward_regions:
  hp_bar: [0.42, 0.92, 0.58, 0.95]
```

---

## 项目结构

```
vision-agent/
├── main.py                                 # CLI 入口
├── gui_app.py                              # PySide6 GUI 入口
├── config.yaml                             # 全局配置
├── profiles/                               # 场景预设配置
│   ├── wzry_5v5.yaml                       # 王者荣耀 (BC)
│   ├── wzry_selfplay.yaml                  # 王者荣耀 (RL)
│   ├── fps_generic.yaml / fps_selfplay.yaml
│   └── desktop.yaml
├── models/                                 # 预训练 ONNX 模型 (可选)
├── vision_agent/
│   ├── core/                               # 视觉编码器 (MobileNetV3)
│   ├── data/                               # 数据采集与训练
│   │   ├── game_recorder.py                # PC 录制 (窗口捕获+键鼠)
│   │   ├── mobile_recorder.py              # 手机录制 (scrcpy+ADB)
│   │   ├── e2e_dataset.py                  # E2E 数据集
│   │   └── e2e_trainer.py                  # E2EMLP 训练器
│   ├── decision/                           # 决策引擎
│   │   ├── e2e_engine.py                   # BC 推理引擎
│   │   ├── dqn_engine.py                   # DQN 推理引擎
│   │   ├── llm_coach.py                    # LLM 教练
│   │   ├── llm_provider.py                 # LLM 供应商抽象
│   │   └── minimax_mcp.py                  # MiniMax MCP 工具
│   ├── rl/                                 # 强化学习
│   │   ├── dqn_agent.py                    # DQN 智能体 (支持 BC 热启动)
│   │   ├── game_env.py                     # 游戏环境 (scrcpy+ADB)
│   │   ├── reward.py                       # 奖励检测 (血条/死亡/胜负)
│   │   ├── self_play.py                    # 自对弈循环
│   │   ├── replay_buffer.py                # 经验回放缓冲区
│   │   └── preset.py                       # 预设加载器
│   ├── workshop/                           # 训练管线
│   │   ├── learning_pipeline.py            # BC 训练 + 伪标签扩展
│   │   ├── model_registry.py               # 模型注册
│   │   └── session.py / scene.py           # 会话与场景管理
│   └── gui/                                # PySide6 GUI
│       ├── main_window.py                  # 主窗口 (3 模式切换)
│       ├── workshop_panel.py               # 训练工坊面板
│       ├── selfplay_panel.py               # Agent 部署面板
│       ├── llm_panel.py                    # LLM 配置面板
│       ├── train_chart.py                  # 训练曲线图表
│       ├── styles.py                       # 深色主题
│       └── widgets.py                      # 通用组件 (折叠区/模型浏览器)
└── requirements.txt
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 视觉编码 | MobileNetV3-Small (torchvision, 576 维特征) |
| 行为克隆 | E2EMLP (PyTorch) |
| 强化学习 | DQN + 经验回放 + ε-greedy |
| LLM 接入 | Claude / OpenAI / DeepSeek / Qwen / MiniMax / Ollama |
| 图像处理 | OpenCV |
| GUI | PySide6 (Qt) |
| 屏幕捕获 | mss |
| 输入模拟 | pynput (PC) / ADB (手机) |
| 手机投屏 | scrcpy |
| 打包分发 | PyInstaller |

---

## RL 自学习参考

RL 自对弈模块参考了 [wzry_ai](https://github.com/myBoris/wzry_ai) 项目，主要改进：

| 方面 | wzry_ai | 本项目 |
|------|---------|--------|
| 起步 | 随机策略 (ε=1.0) | BC 预训练热启动 |
| 视觉 | 2 层 CNN 从零训练 | MobileNetV3 预训练编码器 (576 维) |
| 动作空间 | 8 维并行 (~950 种) | 触控区域映射 (精简离散动作) |
| 奖励 | 纯像素 + OCR | 像素血条 + 灰屏死亡 + 颜色胜负 |
| 网络 | 原始 CNN + 多头 DQN | MobileNetV3 特征 + MLP DQN |

---

## License

MIT
