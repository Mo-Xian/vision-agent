# Changelog

版本更新记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/)。

---

## [v0.2.0] - 2026-03-27

远程 PC 支持（服务端-客户端架构）：训练和 Agent 控制均可跨局域网操作。

### 新功能

- **服务端-客户端架构**：Vision Agent 主机运行中转服务（RemoteHub），远程 PC 运行轻量客户端（RemoteCaptureClient），所有训练和决策在服务端完成
- **远程 PC 录制**：客户端推送画面 + 键鼠事件到中转服务，服务端接收并保存为标准训练格式
- **远程 PC Agent 控制**：Agent 通过中转服务获取远程画面进行决策，操控指令经中转转发到客户端执行
- **客户端自动重连**：网络断开后自动重连，远程 PC 无感恢复
- **客户端独立 EXE**：`RemoteCaptureClient.exe` 单文件打包，远程 PC 无需安装 Python
- **Agent 部署三模式**：手机（ADB 触控）、PC（pynput 键鼠）、远程 PC（中转服务）
- **GUI 中转服务**：录制源和 Agent 部署均可启动/停止中转服务，显示本机地址供客户端连接
- **CLI hub 命令**：`python main.py hub` 启动中转服务并录制

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
