# Vision Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-41cd52.svg)](https://doc.qt.io/qtforpython/)

**端到端行为克隆 + RL 自学习框架，用于游戏 AI 和桌面自动化。**

录制人类操作 → 行为克隆训练 → 伪标签扩展 → RL 自对弈，全流程 GUI 可视化。支持 PC（窗口捕获 + 键鼠）和远程设备（PC 客户端 / 手机 App 通过中转服务连接，支持 Agent 远程操控）。

> **核心思路**：看人怎么玩 → 模仿学会 → 自己练习变强。从零开始，不需要手写规则。

---

## 界面预览

### 训练工坊

场景管理、录制操作、行为克隆训练、自主改进循环、RL 蒸馏，全部在一个面板完成。

![训练工坊](docs/screenshots/mode_train.png)

### Agent 部署

连接手机/PC，选择训练好的模型，Agent 自动检测模型类型并接管游戏操作，实时显示画面。

![Agent 部署](docs/screenshots/mode_agent.png)

### LLM 配置

LLM 连接配置，训练工坊的 LLM Coach、动作发现、伪标签扩展共用此配置。

![LLM 配置](docs/screenshots/mode_llm.png)

---

## 架构：统一学习管线

像人一样学习：**老师教 → 自己学 → 实战练**。一键启动，自动编排全流程。

```
Phase 1 教师教学（一次性）:
  人类录制操作 → 自动检测 → LLM 动作发现 → MobileNetV3 编码
  → 嵌入空间数据增强(高斯噪声 σ=0.02) → MLP 行为克隆训练(早停 patience=20)
  → 实时训练曲线 → LLM 教练诊断

Phase 2 自主改进循环（自动，无需人工介入）:
  Round 0: 处理用户提供的额外视频 → 伪标签扩展
  Round 1~N（默认最多 3 轮）:
    ┌─ 获取视频（按优先级）──────────────────────────┐
    │  1. 用户配置的视频源 URL → yt-dlp 直接下载      │
    │  2. URL 用完 → LLM 分析技能缺口 → 搜索 → 下载   │
    │  3. 都获取不到 → 标记待人工（唯一需人工的场景）   │
    └──────────────────────────────────────────────┘
    → 智能伪标签 — 三级置信度路由:
        模型自信(>85%) → 自动标注
        中等置信度(50-85%) → 余弦去重 → LLM 教练标注
        低置信度(<50%) → 标记待人工审核
    → 合并数据 + 数据增强 → 重新训练
    → 评估: 精度提升 ≥0.5% 则继续，否则停止

Phase 3 自我实践:
  BC 热启动 → 自对弈 (远程设备画面 + 控制指令)
  → DQN ε-greedy 探索 → 奖励检测(血条/击杀/胜负)
  → 经验回放 → Q 网络训练 → 持续改进

部署:
  训练产出 → Agent 接管 → 远程设备控制(PC键鼠/手机触控)
```

**核心特性：初始录制后全自动**。人类只需录制一次操作，后续搜索视频、下载、标注、训练、评估均自动完成。

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

# 自主学习视频下载支持（可选，安装后可自动从 Bilibili/YouTube 下载训练视频）
pip install yt-dlp

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

# 远程录制 — 局域网直连
python main.py hub                                   # 启动中转服务
python main.py hub --port 9876 -o recordings/remote  # 指定端口和输出目录
# 远程设备连接: RemoteCaptureClient.exe ws://本机IP:9876

# 远程录制 — 公网中继（跨网络）
python main.py hub --relay ws://公网服务器:9877 --room myroom
# 远程设备连接: RemoteCaptureClient.exe ws://公网服务器:9877 --room myroom

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
# VisionAgent 主程序打包
build.bat                  # Windows 一键打包
python build_exe.py        # 跨平台打包脚本
python build_exe.py --debug  # 带控制台的调试版
# 产出在 dist/VisionAgent/

# 远程采集客户端打包（单文件 EXE，远程 PC 无需安装 Python）
python build_server.py --onefile
# 产出 dist/RemoteCaptureClient.exe

# 公网中继服务打包（单文件 EXE，部署到公网服务器）
python build_relay.py
# 产出 dist/RelayServer.exe
```

### 自动发布

推送版本 tag 后 GitHub Actions 自动打包并发布到 Releases：

```bash
git tag v1.0.0
git push origin v1.0.0
# → GitHub Actions 自动打包 → Releases 页面下载：
#   VisionAgent-v1.0.0-windows.zip        (主程序)
#   RemoteCaptureClient-v1.0.0-windows.exe (远程采集客户端)
#   RelayServer-v1.0.0-windows.exe         (公网中继服务)
```

---

## 远程设备支持

支持远程录制和 Agent 控制，采用**服务端-客户端架构**，两种连接方式：

```
方式 1: 局域网直连
远程设备 ──ws──> Vision Agent 主机 (WebSocket Server :9876)

方式 2: 公网中继（跨网络）
Vision Agent ──ws──> 公网 Relay Server (:9877) <──ws── 远程设备
                     (纯转发，不解析不缓存)
```

### 公网中继

当 Vision Agent 和远程设备不在同一局域网时，通过轻量中继服务转发：

```bash
# 1. 部署中继到公网服务器（二选一）

# 方式 A: 直接运行（Linux/Windows，仅需 pip install websockets）
pip install websockets
python relay_server.py --port 9877 --token mytoken

# 方式 B: 下载 EXE 运行（Windows 服务器，无需 Python）
RelayServer.exe --port 9877 --token mytoken

# 2. Vision Agent 端
# GUI: 选择「远程设备」→「公网中继」→ 输入中继地址和 token → 启动
# CLI:
python main.py hub --relay ws://公网IP:9877 --room myroom --token mytoken

# 3. 远程设备连接（PC 客户端或 Android App 填同样的中继地址和房间号）
RemoteCaptureClient.exe ws://公网IP:9877 --room myroom --token mytoken
```

**中继特点**: ~160 行纯 Python，唯一依赖 websockets，消息原样转发不解析不缓存，对服务器性能要求极低。支持房间配对（一个 hub + 一个 client）、可选 token 验证、断线通知、自动重连。

### 局域网直连

```bash
# 1. Vision Agent 端
# GUI: 选择「远程设备」→「局域网直连」→ 启动中转服务
# CLI:
python main.py hub --port 9876

# 2. 远程设备连接
RemoteCaptureClient.exe ws://服务端IP:9876
RemoteCaptureClient.exe ws://192.168.1.100:9876 --window "王者荣耀" --fps 15
```

### PC 客户端

从 GitHub Releases 下载 `RemoteCaptureClient.exe`，或用 Python 运行：

| 功能 | 说明 |
|------|------|
| 画面推送 | mss 截屏 → JPEG 编码 → WebSocket 实时推送 |
| 事件推送 | pynput 监听键盘/鼠标事件 → JSON 推送 |
| 控制执行 | 接收 Agent 指令：`key_tap`、`key_press`、`key_release`、`mouse_click`、`mouse_move` |
| 窗口捕获 | `--window "标题"` 模糊匹配指定窗口，不指定则全屏 |
| 自动重连 | 网络断开后 5 秒自动重连 |
| 独立部署 | 单文件 EXE，无需 Python 环境 |

### Android 客户端

无需 USB 调试和 ADB，安装 APK 即用。

1. 用 Android Studio 打开 `android/` 目录，构建安装到手机
2. 输入连接地址（局域网直连填 `ws://192.168.1.100:9876`，公网中继填中继地址 + 房间号）
3. 点击"开始采集" → 授权截屏
4. （可选）开启无障碍服务 → Agent 可远程控制手机

| 功能 | 说明 |
|------|------|
| 画面采集 | MediaProjection API 截屏（系统弹窗授权，无需 root） |
| 控制执行 | AccessibilityService 点击/滑动/按键（设置中开启，无需 root） |
| 通信协议 | WebSocket，与 PC 客户端完全一致，支持直连和中继 |
| 自动重连 | 网络断开后 5 秒自动重连 |
| 要求 | Android 7.0+ |

### 支持场景

- **远程录制训练**：在远程设备操作游戏，画面和操作实时传回生成训练数据
- **Agent 远程操控**：Agent 获取远程画面 → 模型决策 → 控制指令发回远程设备执行
- **跨网络协作**：通过公网中继，本地和远程设备无需同一局域网

---

## 界面说明

界面分为 3 个模式，通过顶部按钮切换：

| 模式 | 用途 | 核心操作 |
|------|------|----------|
| **训练工坊** | 数据录制、BC 训练、自主改进循环、RL 就绪 | 选窗口录制、一键学习、配置视频源、实时训练曲线 |
| **Agent 部署** | 连接手机/PC/远程PC、加载模型、Agent 接管 | 选择目标、选择模型、Agent 接管/停止 |
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
├── relay_server.py                         # 公网中继服务 (独立部署)
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
│   │   ├── remote_hub.py                   # 远程中转服务 (服务端)
│   │   ├── remote_capture_client.py        # 远程采集客户端 (独立 EXE)
│   │   ├── remote_recorder.py              # 远程录制器 (通过 hub 录制)
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
│   │   ├── unified_pipeline.py             # 统一学习管线 (自动编排全流程)
│   │   ├── learning_pipeline.py            # BC 训练 + 伪标签扩展
│   │   ├── video_downloader.py             # 视频下载器 (yt-dlp 封装)
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
├── android/                                # Android 客户端 (无需 ADB)
│   └── app/src/main/java/.../client/
│       ├── MainActivity.kt                 # 主界面 (地址输入+授权+启停)
│       ├── CaptureService.kt               # MediaProjection 截屏服务
│       ├── ControlService.kt               # AccessibilityService 控制执行
│       └── HubConnection.kt                # WebSocket 通信 (OkHttp)
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
| 远程通信 | WebSocket (websockets 库，双向帧/事件/控制，支持直连+公网中继) |
| 视频下载 | yt-dlp (可选，自主学习自动下载) |
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
