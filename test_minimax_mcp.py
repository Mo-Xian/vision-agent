"""测试 MiniMax MCP 工具集成（VLM 图像理解 + 联网搜索）。"""

import base64
import os
import sys

# MiniMax 配置 - 从环境变量读取
API_KEY = os.environ.get("MINIMAX_API_KEY", "")

def test_web_search():
    """测试联网搜索。"""
    print("=" * 50)
    print("测试 1: 联网搜索")
    print("=" * 50)

    from vision_agent.decision.minimax_mcp import MiniMaxMCPTools

    tools = MiniMaxMCPTools(api_key=API_KEY)
    result = tools.web_search("王者荣耀 最新英雄 2026")
    print(f"搜索结果:\n{result}")
    print()


def test_vlm():
    """测试 VLM 图像理解。"""
    print("=" * 50)
    print("测试 2: VLM 图像理解")
    print("=" * 50)

    import cv2
    from vision_agent.decision.minimax_mcp import MiniMaxMCPTools

    tools = MiniMaxMCPTools(api_key=API_KEY)

    # 从视频截取一帧
    video_path = "videos/wzry_gameplay_hd.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, 60000)  # 跳到 60 秒
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取帧")
        return

    # 缩小并编码
    h, w = frame.shape[:2]
    scale = 512 / min(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    image_url = f"data:image/jpeg;base64,{b64}"

    print(f"图片大小: {len(b64) // 1024}KB (base64)")

    result = tools.understand_image(
        "请详细描述这张游戏截图中的画面内容，包括场景、角色、UI元素、血条状态等。",
        image_url,
    )
    print(f"VLM 描述:\n{result}")
    print()


def test_provider_integration():
    """测试通过 Provider 自动调用 MCP VLM。"""
    print("=" * 50)
    print("测试 3: Provider 集成（图片自动转文本）")
    print("=" * 50)

    import cv2
    from vision_agent.decision.llm_provider import create_provider

    provider = create_provider(
        provider_name="minimax",
        api_key=API_KEY,
        model="MiniMax-M2.7",
    )
    print(f"Provider MCP 工具: {'已启用' if provider._mcp_tools else '未启用'}")

    # 准备带图片的消息
    video_path = "videos/wzry_gameplay_hd.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 120000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取帧")
        return

    h, w = frame.shape[:2]
    scale = 512 / min(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "这是一张王者荣耀游戏截图，请分析当前局势并推荐下一步动作。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }]

    print("调用 MiniMax-M2.7（图片将通过 MCP VLM 预处理）...")
    response = provider.chat(
        messages=messages,
        system="你是一个游戏决策助手。",
        max_tokens=512,
    )
    print(f"M2.7 回复:\n{response.text}")
    print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    tests = sys.argv[1:] if len(sys.argv) > 1 else ["search", "vlm", "provider"]

    if "search" in tests:
        test_web_search()
    if "vlm" in tests:
        test_vlm()
    if "provider" in tests:
        test_provider_integration()
