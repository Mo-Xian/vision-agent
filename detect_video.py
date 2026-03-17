"""对视频文件运行 YOLO 检测，输出带标注的结果视频。

用法:
    python detect_video.py --model your_model.pt --video input.mp4
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO 视频检测")
    parser.add_argument("--model", default="yolov8n.pt", help="模型路径")
    parser.add_argument("--video", default="input.mp4", help="视频路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--output", default="runs/detect_output", help="输出目录")
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    print(f"开始检测: {args.video}")
    model(args.video, save=True, conf=args.conf, project=args.output, name="result", exist_ok=True)

    print(f"检测完成! 结果保存在: {args.output}/result/")


if __name__ == "__main__":
    main()
