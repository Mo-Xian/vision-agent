"""强化学习自学习模块：行为克隆预训练 + DQN 微调。

核心组件：
  - GameEnvironment: 游戏环境封装（截图→动作执行→奖励→下一状态）
  - RewardDetector:  奖励检测器（血条/击杀/死亡/胜负识别）
  - DQNAgent:        DQN 智能体（支持从 BC 模型热启动）
  - SelfPlayLoop:    自对弈循环（采集+训练双线程）
"""
