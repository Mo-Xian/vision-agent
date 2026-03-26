# Vision Agent

## 基本信息

| 属性 | 值 |
|------|-----|
| 项目名 | vision-agent |
| 语言 | Python 3.10+ |
| 核心依赖 | PyTorch, torchvision (MobileNetV3), OpenCV, PySide6, openai |
| 状态 | 活跃开发 |

## 项目定位

端到端视频学习框架：给视频 → LLM 分析/标注 → 视觉编码 + MLP 训练 → 决策模型。

## 架构

```
视频 → LLM视觉分析(场景/动作识别)
     → LLM标注关键帧(决策数据)
     → MobileNetV3视觉编码(576维嵌入)
     → 标签传播(数据放大)
     → MLP训练(行为克隆)
     → 策略梯度RL(可选强化)
     → 产出模型 + Profile
```

## 核心文件

| 文件 | 用途 |
|------|------|
| `main.py` | CLI 入口（learn/eval 子命令） |
| `gui_app.py` | PySide6 GUI 入口 |
| `eval_model.py` | 模型评估工具（标注视频/训练曲线/统计） |
| `config.yaml` | 全局配置 |
| `vision_agent/core/vision_encoder.py` | MobileNetV3 视觉编码器 |
| `vision_agent/core/keyframe.py` | 关键帧采样器 |
| `vision_agent/data/auto_annotator.py` | LLM 自动标注器 |
| `vision_agent/data/e2e_dataset.py` | E2E 数据集 |
| `vision_agent/data/e2e_trainer.py` | E2E MLP 训练器 |
| `vision_agent/decision/base.py` | Action + DecisionEngine ABC |
| `vision_agent/decision/e2e_engine.py` | E2E 推理引擎 |
| `vision_agent/decision/llm_provider.py` | LLM 供应商抽象层 |
| `vision_agent/decision/minimax_mcp.py` | MiniMax MCP 工具（VLM+搜索） |
| `vision_agent/workshop/learning_pipeline.py` | 学习管线（分析→标注→训练→RL） |
| `vision_agent/workshop/video_analyzer.py` | LLM 视频分析器 |
| `vision_agent/workshop/scene.py` | 场景管理 |
| `vision_agent/workshop/model_registry.py` | 模型注册中心 |
| `vision_agent/workshop/session.py` | 训练会话 |
| `vision_agent/gui/main_window.py` | GUI 主窗口（工坊+LLM设置） |
| `vision_agent/gui/workshop_panel.py` | 训练工坊面板 |
| `vision_agent/gui/llm_panel.py` | LLM 配置面板 |

## 开发命令

```bash
# GUI 模式
python gui_app.py

# CLI 学习
python main.py learn video1.mp4 --provider minimax --model MiniMax-M2.7

# CLI 评估
python main.py eval runs/workshop/exp1/model --mode stats
python main.py eval runs/workshop/exp1/model --video video.mp4 --mode video

# 测试 MiniMax MCP
python test_minimax_mcp.py
```

## 数据管线

LLM 自动标注 → 视觉编码 → 标签传播 → MLP 训练 → RL 强化

1. **LLM 视觉分析**：截图发给 LLM，识别场景和动作集
2. **LLM 标注关键帧**：LLM 看截图判断当前应执行什么动作 → JSONL
3. **视觉编码**：MobileNetV3-Small 将帧编码为 576 维向量
4. **标签传播**：关键帧标签扩展到相邻帧（~10x 数据放大）
5. **MLP 训练**：576 → 256 → 128 → N 动作分类
6. **RL 强化**：策略梯度微调（可选）

训练产出：`model.pt` + `model.meta.json`

## 约定

- API key 通过环境变量读取，不写入代码或配置
- 模型文件 `*.pt`、训练数据 `datasets/`、训练输出 `runs/` 通过 `.gitignore` 排除
- 视频文件 `*.mp4` 不纳入版本控制

# currentDate
Today's date is 2026-03-26.
