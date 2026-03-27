"""MiniMax MCP 工具集成：图像理解 + 联网搜索。

MiniMax-M2.7 是纯文本模型，需要通过 MCP 工具实现多模态能力。
本模块直接调用 MiniMax Coding Plan API（与 minimax-coding-plan-mcp 相同的后端），
无需运行 MCP Server 子进程。

API 端点（api.minimaxi.com）:
  - 图像理解: POST /v1/coding_plan/vlm  {prompt, image_url}
  - 联网搜索: POST /v1/coding_plan/search  {q}

注意：Token Plan VLM 有周期性调用配额（错误码 2056），
超限后需等待下一个周期恢复。本模块会自动重试和限速。
"""

import base64
import logging
import time

import requests

logger = logging.getLogger(__name__)

# MiniMax MCP API 主机（与 chat API 的 api.minimax.chat 不同）
MINIMAX_MCP_HOST = "https://api.minimaxi.com"


class MiniMaxMCPTools:
    """MiniMax Coding Plan MCP 工具客户端。

    自动限速和重试，应对 Token Plan 的周期性配额限制（2056）。
    """

    # VLM 调用间隔（秒），避免触发频率限制
    VLM_MIN_INTERVAL = 2.0
    # 配额超限时的等待时间（秒）
    VLM_QUOTA_WAIT = 60.0
    # 最大重试次数
    VLM_MAX_RETRIES = 3

    def __init__(self, api_key: str, api_host: str = MINIMAX_MCP_HOST):
        self._api_key = api_key
        self._api_host = api_host.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._last_vlm_call = 0.0

    def _vlm_rate_limit(self):
        """VLM 调用限速。"""
        elapsed = time.time() - self._last_vlm_call
        if elapsed < self.VLM_MIN_INTERVAL:
            time.sleep(self.VLM_MIN_INTERVAL - elapsed)
        self._last_vlm_call = time.time()

    def understand_image(self, prompt: str, image_b64_url: str) -> str:
        """调用 VLM 理解图片内容，带自动重试。

        Args:
            prompt: 对图片的分析要求
            image_b64_url: data:image/jpeg;base64,... 格式的图片

        Returns:
            VLM 返回的文本描述
        """
        payload = {
            "prompt": prompt,
            "image_url": image_b64_url,
        }

        for attempt in range(self.VLM_MAX_RETRIES):
            self._vlm_rate_limit()
            try:
                resp = self._session.post(
                    f"{self._api_host}/v1/coding_plan/vlm",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                base_resp = data.get("base_resp", {})
                status_code = base_resp.get("status_code", 0)

                if status_code == 0:
                    content = data.get("content", "")
                    return content if content else "[图像理解: 无返回内容]"

                err = base_resp.get("status_msg", "unknown error")

                # 2056 = Token Plan 配额超限，等待后重试
                if status_code == 2056 and attempt < self.VLM_MAX_RETRIES - 1:
                    wait = self.VLM_QUOTA_WAIT * (attempt + 1)
                    logger.warning(f"VLM 配额超限，等待 {wait:.0f}s 后重试 ({attempt+1}/{self.VLM_MAX_RETRIES})")
                    time.sleep(wait)
                    continue

                # 1002 = 请求频率超限，短暂等待后重试
                if status_code == 1002 and attempt < self.VLM_MAX_RETRIES - 1:
                    logger.warning(f"VLM 频率限制，等待 10s 后重试")
                    time.sleep(10)
                    continue

                logger.error(f"MiniMax VLM 错误 [{status_code}]: {err}")
                return f"[图像理解失败 [{status_code}]: {err}]"

            except Exception as e:
                logger.error(f"MiniMax VLM 请求失败: {e}")
                if attempt < self.VLM_MAX_RETRIES - 1:
                    time.sleep(5)
                    continue
                return f"[图像理解失败: {e}]"

        return "[图像理解失败: 重试次数耗尽]"

    def web_search(self, query: str) -> list[dict]:
        """联网搜索。

        Args:
            query: 搜索关键词

        Returns:
            搜索结果列表，每项含 url/title/snippet 字段
        """
        payload = {"q": query}
        try:
            resp = self._session.post(
                f"{self._api_host}/v1/coding_plan/search",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            base_resp = data.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                return []

            organic = data.get("organic", [])
            if not organic:
                return []

            results = []
            for item in organic[:5]:
                results.append({
                    "url": item.get("link", item.get("url", "")),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                })
            return results

        except Exception as e:
            logger.error(f"MiniMax 搜索请求失败: {e}")
            return f"[搜索失败: {e}]"


def preprocess_messages_with_vlm(
    messages: list[dict],
    mcp_tools: MiniMaxMCPTools,
    vlm_prompt: str = "请详细描述这张游戏截图中的画面内容，包括场景、角色、UI元素、血条状态等。",
    max_vlm_calls: int = 3,
) -> list[dict]:
    """将消息中的图片通过 VLM 转为文本描述。

    遍历 messages，找到 image_url 类型的内容块，
    调用 MiniMax VLM 获取文本描述后替换。
    为节省配额，当图片超过 max_vlm_calls 张时，
    均匀选取部分图片调用 VLM，其余标记为「参考上下文图片」。

    Args:
        messages: 原始消息列表（可能包含 image_url）
        mcp_tools: MiniMax MCP 工具实例
        vlm_prompt: VLM 分析提示词
        max_vlm_calls: 单次消息中最多调用 VLM 的图片数

    Returns:
        处理后的纯文本消息列表
    """
    # 先统计总图片数
    image_indices = []  # (msg_idx, item_idx)
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for ii, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_indices.append((mi, ii))

    total_images = len(image_indices)

    # 决定哪些图片调用 VLM
    if total_images <= max_vlm_calls:
        vlm_set = set(range(total_images))
    else:
        # 均匀选取
        step = max(1, total_images // max_vlm_calls)
        vlm_set = set(i * step for i in range(max_vlm_calls) if i * step < total_images)
        # 确保最后一张也包含
        vlm_set.add(total_images - 1)

    logger.info(f"[MCP VLM] {total_images} 张图片, 调用 VLM {len(vlm_set)} 次")

    # 处理消息
    img_counter = 0
    processed = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            processed.append(msg)
            continue

        new_content = []
        for ii, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:") and img_counter in vlm_set:
                    description = mcp_tools.understand_image(vlm_prompt, url)
                    new_content.append({
                        "type": "text",
                        "text": f"[图片 {img_counter+1}/{total_images} 内容描述]: {description}",
                    })
                else:
                    new_content.append({
                        "type": "text",
                        "text": f"[图片 {img_counter+1}/{total_images}: 未分析，请参考已分析图片的上下文推断]",
                    })
                img_counter += 1
            else:
                new_content.append(item)

        processed.append({"role": msg["role"], "content": new_content})

    return processed
