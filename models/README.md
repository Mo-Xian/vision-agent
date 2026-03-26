# 预训练模型

## wzry_ai ONNX 模型

王者荣耀自对弈需要以下 ONNX 模型（来自 [wzry_ai](https://github.com/myBoris/wzry_ai) 项目）：

| 文件 | 用途 | 必需 |
|------|------|------|
| `start.onnx` | 对局开始检测 | 可选（无则跳过等待） |
| `death.onnx` | 死亡状态检测 | 可选（无则用像素启发式） |

### 获取方式

```bash
# 方式 1：从 wzry_ai 仓库下载
git clone https://github.com/myBoris/wzry_ai.git --depth 1
cp wzry_ai/models/start.onnx models/
cp wzry_ai/models/death.onnx models/

# 方式 2：仅下载模型文件
# 从 https://github.com/myBoris/wzry_ai/tree/main/models 手动下载
```

### 模型说明

- **start.onnx**: YOLO 格式目标检测模型，输入 640x640，检测 "started" 类别
- **death.onnx**: YOLO 格式目标检测模型，输入 640x640，检测 "death" 类别
- 两个模型均支持 CPU 和 CUDA 推理
- 需要安装 `onnxruntime` 或 `onnxruntime-gpu`

### 无模型时的行为

如果模型文件不存在，系统会自动回退到像素级启发式检测：
- 对局开始：跳过等待，直接开始采集
- 死亡检测：灰屏饱和度分析（连续 3 帧低饱和度 → 判定死亡）
- 胜负检测：颜色特征分析（金色 → 胜利，暗灰 → 失败）
