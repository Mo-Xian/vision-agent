# Vision Agent

## 基本信息

| 属性 | 值 |
|------|-----|
| 项目名 | vision-agent |
| 语言 | Python 3.10+ |
| 核心依赖 | PyTorch, torchvision (MobileNetV3), OpenCV, PySide6, openai, mss, pynput |
| 状态 | 活跃开发 |

## 项目定位

端到端行为克隆 + RL 自学习框架：录制人类游戏操作 → 行为克隆 → 伪标签扩展 → RL 自对弈。

支持 PC（窗口捕获+键鼠）和手机（scrcpy + ADB 触控）。

## 架构

```
统一学习管线（UnifiedPipeline）— 模拟人类学习过程:

Phase 1 教师教学:
  检测输入(人类录制 vs 纯视频) → BC 行为克隆训练 → 初始策略

Phase 2 自主学习:
  2a. 技能差距分析 — LLM 教练评估缺什么技能
  2b. 在线搜索 — 搜索相关教学视频资源
  2c. 智能伪标签 — 对新视频分帧处理:
      模型自信(>85%) → 直接标注
      中等置信度 → 询问 LLM 教练
      LLM 也不确定 → 标记待人工补充

Phase 3 自我实践:
  BC 模型热启动 → GameEnvironment(scrcpy截屏+ADB操作)
  → DQN Agent(ε-greedy) → RewardDetector(血条/击杀/胜负)
  → 经验回放 → Q网络训练 → 持续改进
```

## 核心文件

| 文件 | 用途 |
|------|------|
| `main.py` | CLI 入口（record/mobile/learn-bc/expand/self-play/eval） |
| `gui_app.py` | PySide6 GUI 入口 |
| `vision_agent/core/vision_encoder.py` | MobileNetV3 视觉编码器 |
| `vision_agent/data/game_recorder.py` | PC 录制器（窗口捕获+键鼠+流式保存） |
| `vision_agent/data/mobile_recorder.py` | 手机录制器（scrcpy+ADB 触摸采集） |
| `vision_agent/data/e2e_dataset.py` | E2E 数据集 |
| `vision_agent/data/e2e_trainer.py` | E2E MLP 训练器 |
| `vision_agent/decision/base.py` | Action + DecisionEngine ABC |
| `vision_agent/decision/e2e_engine.py` | E2E 推理引擎 |
| `vision_agent/decision/llm_coach.py` | LLM 教练（动作发现/诊断） |
| `vision_agent/rl/game_env.py` | 游戏 RL 环境（scrcpy截屏+ADB操作） |
| `vision_agent/rl/reward.py` | 奖励检测器（血条/死亡/胜负） |
| `vision_agent/rl/dqn_agent.py` | DQN 智能体（支持 BC 热启动） |
| `vision_agent/rl/self_play.py` | 自对弈循环（采集+训练双线程） |
| `vision_agent/rl/preset.py` | 自对弈预设加载器（王者荣耀内置） |
| `vision_agent/rl/replay_buffer.py` | 经验回放缓冲区 |
| `vision_agent/workshop/unified_pipeline.py` | 统一学习管线（教师教学→自主学习→自我实践） |
| `vision_agent/workshop/learning_pipeline.py` | BC 学习管线（训练+伪标签扩展） |
| `vision_agent/gui/main_window.py` | GUI 主窗口 |
| `vision_agent/gui/workshop_panel.py` | 训练工坊面板 |
| `profiles/wzry_5v5.yaml` | 王者荣耀 Profile（含触控区域） |
| `profiles/wzry_selfplay.yaml` | 王者荣耀自对弈预设（完整配置） |

## 开发命令

```bash
# GUI 模式
python gui_app.py

# PC 录制（F9 暂停/恢复，Ctrl+C 停止）
python main.py record --output recordings/session1 --fps 10
python main.py record --window "王者荣耀"

# 手机录制（scrcpy + ADB）
python main.py mobile --check                        # 检查环境
python main.py mobile --game moba                    # MOBA 预设
python main.py mobile --zones touch_zones.json       # 自定义区域

# 行为克隆训练
python main.py learn-bc recordings/session1 recordings/session2

# 伪标签扩展
python main.py expand runs/workshop/exp1/model video1.mp4 video2.mp4

# RL 自对弈
python main.py self-play --preset wzry                          # 王者荣耀预设
python main.py self-play --preset wzry --bc-model runs/.../model  # BC 热启动
python main.py self-play --game moba                            # 通用 MOBA

# 评估模型
python main.py eval runs/workshop/exp1/model --mode stats
```

## RL 自学习流程

参考 [wzry_ai](https://github.com/myBoris/wzry_ai) 项目，但有关键改进：

| 方面 | wzry_ai | 本项目 |
|------|---------|--------|
| 起步 | 随机策略 (ε=1.0) | BC 预训练热启动 |
| 视觉 | 2层CNN从零训练 | MobileNetV3 预训练编码器 (576维) |
| 动作空间 | 8维并行 (~950种) | 触控区域映射 (精简离散动作) |
| 奖励 | 纯像素+OCR | 像素血条+灰屏死亡+颜色胜负 |
| 网络 | 原始CNN+多头DQN | MobileNetV3特征+MLP DQN |

王者荣耀预设内置了 wzry_ai 的按钮坐标和奖励区域配置。

## 约定

- API key 通过环境变量读取，不写入代码或配置
- 模型文件 `*.pt`、训练数据 `datasets/`、训练输出 `runs/`、录制 `recordings/` 通过 `.gitignore` 排除
- 视频文件 `*.mp4` 不纳入版本控制
- 硬件限制：CPU-only，使用 MobileNetV3-Small 作为视觉编码器
