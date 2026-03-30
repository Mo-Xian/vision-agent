# Changelog

版本更新记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/)。

---

## [v0.3.0] - 2026-03-30

统一远程架构：公网中继 + RemoteHub 全链路，彻底移除 scrcpy/ADB 依赖。

### 新功能

- **公网中继服务**：`relay_server.py` 轻量房间转发（~160 行，仅依赖 websockets），部署到公网服务器后双方主动连接，消息原样转发不解析不缓存
- **RelayServer EXE 打包**：CI 自动构建 `RelayServer.exe`，Windows 服务器无需 Python
- **RemoteHub 中继模式**：新增 `relay_url`/`room_id` 参数，支持局域网直连和公网中继两种模式
- **RemoteHub 触控接口**：新增 `send_tap()`、`send_swipe()`、`screen_size`，供 RL 环境调用
- **GUI 公网中继选项**：训练工坊和 Agent 部署面板新增「局域网直连 / 公网中继」切换，填入中继地址和房间号即可跨网连接
- **Android 客户端**：原生 App（MediaProjection 截屏 + AccessibilityService 控制），无需 USB 调试和 ADB
- **Android APK CI 构建**：GitHub Actions 自动构建 debug/release APK 并发布到 Releases
- **Android 中继支持**：App 支持填写房间 ID 和 Token，可通过公网中继连接
- **PC 客户端中继支持**：`RemoteCaptureClient` 新增 `--room`/`--token` 参数
- **CLI hub 中继参数**：`python main.py hub --relay ws://server:9877 --room xxx`
- **CLI self-play 远程连接**：`python main.py self-play --preset wzry --relay ws://server:9877 --room xxx`

### 变更

- **RL 自对弈重构为 RemoteHub 模式**：`GameEnvironment` 通过 RemoteHub 获取画面和发送触控指令，不再依赖 scrcpy 截屏和 ADB 命令
- **SelfPlayLoop / UnifiedPipeline**：`device_serial` 参数替换为 `hub`（RemoteHub 实例）
- **DQNEngine 精简为 PC 键盘模式**：移除 ADB 触控执行，远程触控改由 RemoteHub 转发
- **GUI 简化为两种模式**：录制源和部署目标从 3 选项（PC/手机/远程PC）简化为 2 选项（PC/远程设备）
- **CI 拆分为三阶段并行**：build-windows + build-android + release
- 删除 `mobile_recorder.py`（scrcpy+ADB 手机录制器，~750 行）
- 移除 CLI `mobile` 子命令及 self-play 的 `--game`/`--device` 参数
- 移除 `is_mobile_source()`、`is_mobile_target()` 及相关 ADB 管理代码

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
- PySide6 GUI, OpenCV, mss, pynput, websockets
- PyInstaller 打包
