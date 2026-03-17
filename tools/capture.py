"""游戏画面截图采集工具。

按 F1 截取当前屏幕保存到指定目录，用于后续标注训练。
按 ESC 退出。

用法:
    python tools/capture.py --output datasets/honor_of_kings/train/images
    python tools/capture.py --output datasets/honor_of_kings/val/images
"""

import argparse
import time
from pathlib import Path

import cv2
import mss
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Game screenshot capture tool")
    parser.add_argument(
        "--output", "-o",
        default="datasets/honor_of_kings/train/images",
        help="Output directory for captured images",
    )
    parser.add_argument("--monitor", "-m", type=int, default=1, help="Monitor number")
    parser.add_argument("--prefix", default="frame", help="Filename prefix")
    parser.add_argument(
        "--auto", type=float, default=0,
        help="Auto-capture interval in seconds (0 = manual mode, press F1)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing = list(output_dir.glob(f"{args.prefix}_*.jpg"))
    counter = len(existing)

    sct = mss.mss()
    mon = sct.monitors[args.monitor]

    print(f"Screenshot Capture Tool")
    print(f"  Output:  {output_dir.resolve()}")
    print(f"  Monitor: {args.monitor} ({mon['width']}x{mon['height']})")
    print(f"  Mode:    {'Auto (%.1fs)' % args.auto if args.auto > 0 else 'Manual (press S to save)'}")
    print(f"  Existing: {counter} images")
    print()
    print("  S = Save screenshot")
    print("  Q / ESC = Quit")
    print()

    last_capture = 0
    window_name = "Capture Preview (press S to save, Q to quit)"

    while True:
        img = sct.grab(mon)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Show preview (resized)
        h, w = frame.shape[:2]
        scale = min(960 / w, 540 / h)
        preview = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Add info text
        info = f"Saved: {counter} | Press S to capture"
        cv2.putText(preview, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(30) & 0xFF

        now = time.time()
        should_save = False

        # Manual: press 's'
        if key == ord("s"):
            should_save = True

        # Auto mode
        if args.auto > 0 and (now - last_capture) >= args.auto:
            should_save = True

        if should_save:
            counter += 1
            filename = f"{args.prefix}_{counter:05d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            last_capture = now
            print(f"  [{counter}] Saved: {filepath.name}")

        # Quit
        if key in (ord("q"), 27):  # q or ESC
            break

    cv2.destroyAllWindows()
    sct.close()
    print(f"\nDone. Total {counter} images in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
