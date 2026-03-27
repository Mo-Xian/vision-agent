# Changelog

版本更新记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/)。

---

## [v0.2.0] - 2026-03-27

新增远程 PC 录制功能，支持局域网内跨 PC 采集游戏画面和操作。

### 新功能

- **远程 PC 录制**：WebSocket 实时推送画面 + 键鼠事件，本机接收并保存为标准训练格式
- **采集服务端** (`remote_capture_server.py`)：远程 PC 轻量运行，仅需 mss + pynput + websockets + opencv
- **远程录制器** (`remote_recorder.py`)：本机接收端，输出与本地录制完全一致的 recording.mp4 + actions.jsonl
- **GUI 远程录制源**：录制源新增"远程 PC"选项，支持地址配置和连接测试
- **CLI serve/remote 命令**：`python main.py serve` 启动采集服务，`python main.py remote <IP>` 连接录制

---

## [v0.1.1] - 2026-03-27

训练流程审计修复，确保全管线数据增强一致性。

### 修复

- 修复 `_self_study()` 伪标签样本缺少高斯噪声数据增强，导致自主学习轮次数据多样性不足
- 修复 `_rl_and_distill()` RL 经验注入 BC 数据集时缺少数据增强
- 修复 `LearningPipeline` 中 VisionEncoder 每次调用重复创建，改为懒加载缓存复用

---

## [v0.1.0] - 2026-03-27

首个公开版本，实现端到端行为克隆 + RL 自学习完整框架。

### 新功能

- **统一学习管线**：教师教学 → 自主改进循环 → RL 蒸馏，一键启动全自动编排
- **自主改进循环**：BC 训练后自动搜索视频 → 下载 → 伪标签 → 重训练（默认 3 轮）
- **RL→BC 经验蒸馏**：自对弈高奖励经验自动回流 BC 重训练，统一为 E2EMLP 模型
- **视频源可配置**：用户 URL 优先 → LLM 搜索补充 → 人工兜底
- **Agent 部署双模式**：手机（scrcpy + ADB 触控）和 PC（窗口捕获 + 键鼠）
- **Agent 引擎自动检测**：根据 model.meta.json 自动选择 E2EEngine 或 DQNEngine
- **手机录制**：scrcpy 投屏 + ADB 触摸采集
- **RL 自对弈**：DQN + 经验回放 + BC 热启动，支持王者荣耀/FPS 预设
- **LLM 多供应商**：Claude / OpenAI / DeepSeek / Qwen / MiniMax / Ollama
- **MiniMax MCP 集成**：VLM 图像理解 + 搜索 API
- **模型浏览器**：扫描所有训练产出，按准确率排序
- **场景管理**：创建/切换/删除场景，训练历史持久化
- **GitHub Actions 自动打包**：推送 tag 自动生成 EXE 并发布到 Releases

### 修复

- 修复 selfplay_episodes 永远为 0 导致 RL 阶段从不触发
- 修复 coach_advice 被 _coach_summary 覆盖丢失 rl_ready 配置
- 修复 VisionEncoder 重复创建浪费内存
- 修复伪标签扩展缺少图表回调和数据增强
- 修复 LLM 配置重启后丢失
- 修复 GUI 进度标签缺少新阶段（self_study/searching_videos 等）

### 技术栈

- Python 3.10+, PyTorch 2.0+, MobileNetV3-Small (576 维特征)
- PySide6 GUI, OpenCV, mss, pynput, scrcpy + ADB
- PyInstaller 打包
