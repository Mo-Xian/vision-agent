"""网络资源搜索与下载：用 LLM 规划搜索策略，自动获取视频/图片。

支持的视频源:
  - Bilibili（B站）搜索与下载
  - 直接 URL 下载（任意视频链接）
  - yt-dlp 支持的所有站点（可选）
"""

import json
import logging
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
}


class ResourceFetcher:
    """LLM 驱动的网络资源搜索与下载。

    流程: 用户兴趣描述 → LLM 规划搜索关键词 → 搜索获取资源 → 下载到本地
    """

    def __init__(self, llm_provider=None, output_dir: str = "data/fetched",
                 on_log=None):
        self._llm = llm_provider
        self._output_dir = Path(output_dir)
        self._on_log = on_log
        self._stop = False

    def stop(self):
        self._stop = True

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass

    # ── LLM 搜索规划 ──

    def plan_search(self, interest: str, resource_type: str = "video") -> dict:
        """让 LLM 根据用户兴趣规划搜索策略。"""
        prompt = f"""用户对以下场景感兴趣，需要搜索相关的{resource_type}素材来训练视觉AI模型。

用户兴趣描述: {interest}

请生成搜索策略，返回 JSON（不要其他文字）:
{{
    "keywords": ["搜索关键词1", "搜索关键词2", ...],
    "keywords_en": ["english keyword1", "english keyword2", ...],
    "description": "场景描述（用于YOLO检测目标说明）",
    "suggested_actions": ["动作1", "动作2", ...],
    "action_descriptions": {{"动作1": "动作描述", ...}}
}}

要求:
- keywords: 3-5个中文搜索关键词，适合在B站等中文平台搜索
- keywords_en: 对应的英文关键词
- description: 简短场景描述
- suggested_actions: 4-6个适合该场景的决策动作（含 idle）
- action_descriptions: 每个动作的简短说明"""

        if self._llm:
            try:
                response = self._llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system="你是搜索策略规划助手。只返回 JSON。",
                    max_tokens=1024,
                )
                text = response.text or ""
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0]
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0]
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except Exception as e:
                self._log(f"LLM 规划失败: {e}")

        return {
            "keywords": [interest],
            "keywords_en": [interest],
            "description": interest,
            "suggested_actions": ["attack", "defend", "move", "skill", "idle"],
            "action_descriptions": {},
        }

    # ── Bilibili 搜索 ──

    def search_bilibili(self, keywords: list[str], max_results: int = 5) -> list[dict]:
        """通过 Bilibili API 搜索视频。"""
        results = []
        for kw in keywords:
            if self._stop:
                break
            self._log(f"B站搜索: {kw}")
            try:
                resp = requests.get(
                    "https://api.bilibili.com/x/web-interface/search/type",
                    params={
                        "keyword": kw,
                        "search_type": "video",
                        "page": 1,
                        "pagesize": max_results,
                    },
                    headers=_HEADERS,
                    timeout=10,
                )
                data = resp.json()
                for item in (data.get("data", {}).get("result", []) or []):
                    bvid = item.get("bvid", "")
                    if not bvid:
                        continue
                    # 清理 HTML 标签
                    title = re.sub(r"<.*?>", "", item.get("title", ""))
                    results.append({
                        "type": "video",
                        "url": f"https://www.bilibili.com/video/{bvid}",
                        "bvid": bvid,
                        "title": title,
                        "duration": item.get("duration", ""),
                        "source": "bilibili",
                        "keyword": kw,
                    })
            except Exception as e:
                self._log(f"B站搜索失败 ({kw}): {e}")

        # 去重
        seen = set()
        unique = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)
        results = unique

        self._log(f"B站共找到 {len(results)} 个视频")
        return results

    # ── Bilibili 下载 ──

    def download_bilibili(self, bvid: str, filename: str = "") -> str | None:
        """下载 Bilibili 视频。"""
        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 获取 cid
            resp = requests.get(
                f"https://api.bilibili.com/x/player/pagelist",
                params={"bvid": bvid},
                headers=_HEADERS,
                timeout=10,
            )
            pages = resp.json().get("data", [])
            if not pages:
                self._log(f"获取 cid 失败: {bvid}")
                return None
            cid = pages[0]["cid"]

            # 获取视频流 URL
            resp = requests.get(
                "https://api.bilibili.com/x/player/playurl",
                params={"bvid": bvid, "cid": cid, "qn": 16, "fnval": 1},
                headers=_HEADERS,
                timeout=10,
            )
            play_data = resp.json().get("data", {})
            durl = play_data.get("durl", [])
            if not durl:
                self._log(f"获取视频流失败: {bvid}")
                return None

            video_url = durl[0]["url"]

            # 下载
            if not filename:
                filename = f"{bvid}.mp4"
            path = out_dir / filename

            self._log(f"下载B站视频: {bvid}")
            resp = requests.get(
                video_url,
                headers={**_HEADERS, "Referer": f"https://www.bilibili.com/video/{bvid}"},
                stream=True,
                timeout=30,
            )
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (1024 * 512) == 0:
                        pct = downloaded / total * 100
                        self._log(f"  下载进度: {pct:.0f}%")

            self._log(f"下载完成: {filename} ({downloaded // 1024}KB)")
            return str(path)

        except Exception as e:
            self._log(f"B站下载失败 ({bvid}): {e}")
            return None

    # ── 通用 URL 下载 ──

    def download_video_url(self, url: str, filename: str = "") -> str | None:
        """直接下载视频 URL（支持任意 HTTP 视频链接）。"""
        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            # 从 URL 提取文件名
            name = url.split("/")[-1].split("?")[0]
            if not name.endswith((".mp4", ".avi", ".mkv", ".webm")):
                name = f"video_{int(time.time())}.mp4"
            filename = name
        path = out_dir / filename

        try:
            self._log(f"下载视频: {url[:80]}...")
            resp = requests.get(url, headers=_HEADERS, stream=True, timeout=30)
            resp.raise_for_status()

            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)

            size_kb = path.stat().st_size // 1024
            self._log(f"下载完成: {filename} ({size_kb}KB)")
            return str(path)
        except Exception as e:
            self._log(f"视频下载失败: {e}")
            return None

    # ── yt-dlp 通用下载（可选） ──

    def download_with_ytdlp(self, url: str, filename: str = "") -> str | None:
        """用 yt-dlp 下载视频（支持 B站/YouTube 等多平台）。"""
        try:
            import yt_dlp
        except ImportError:
            self._log("[提示] yt-dlp 未安装，尝试直接下载")
            return self.download_video_url(url, filename)

        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "worst[ext=mp4]/worst",
            "outtmpl": str(out_dir / (filename or "%(title)s.%(ext)s")),
            "max_filesize": 100 * 1024 * 1024,
            "socket_timeout": 30,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = ydl.prepare_filename(info)
                self._log(f"下载完成: {Path(path).name}")
                return path
        except Exception as e:
            self._log(f"yt-dlp 下载失败: {e}")
            return None

    # ── 图片搜索 ──

    def search_images_bing(self, keywords: list[str], max_per_keyword: int = 10) -> list[dict]:
        """通过 Bing 图片搜索获取图片 URL。"""
        results = []
        for kw in keywords:
            if self._stop:
                break
            self._log(f"搜索图片: {kw}")
            try:
                resp = requests.get(
                    "https://www.bing.com/images/search",
                    params={"q": kw, "form": "HDRSC2", "first": 1},
                    headers=_HEADERS,
                    timeout=10,
                )
                # 从页面中提取图片 URL
                urls = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', resp.text)
                for url in urls[:max_per_keyword]:
                    results.append({
                        "type": "image",
                        "url": url,
                        "title": kw,
                        "source": "bing",
                        "keyword": kw,
                    })
            except Exception as e:
                self._log(f"图片搜索失败 ({kw}): {e}")

        self._log(f"共找到 {len(results)} 张图片")
        return results

    def download_image(self, url: str, filename: str = "") -> str | None:
        """下载图片到本地。"""
        out_dir = self._output_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = f"img_{int(time.time() * 1000)}.jpg"
        path = out_dir / filename

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return str(path)
        except Exception as e:
            self._log(f"图片下载失败: {e}")
            return None

    # ── 统一入口 ──

    def fetch(self, interest: str, resource_type: str = "video",
              max_results: int = 5, source: str = "bilibili",
              progress_callback=None) -> dict:
        """完整的搜索+下载流程。

        Args:
            interest: 用户兴趣描述
            resource_type: "video" 或 "image"
            max_results: 每个关键词的最大搜索结果数
            source: 视频源 "bilibili" / "url" / "ytdlp"
            progress_callback: (phase, pct) 回调
        """
        self._stop = False
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 1. LLM 规划
        self._log("Step 1/3: LLM 规划搜索策略...")
        plan = self.plan_search(interest, resource_type)
        self._log(f"搜索关键词: {plan['keywords']}")
        self._log(f"建议动作: {plan['suggested_actions']}")
        if progress_callback:
            progress_callback("planning", 0.1)

        if self._stop:
            return {"plan": plan, "resources": [], "local_files": []}

        # 2. 搜索
        self._log("Step 2/3: 搜索网络资源...")
        keywords = plan.get("keywords", [])

        if resource_type == "image":
            resources = self.search_images_bing(keywords, max_per_keyword=max_results)
        elif source == "bilibili":
            resources = self.search_bilibili(keywords, max_results=max_results)
        else:
            resources = []  # URL 模式不搜索，直接由用户或 LLM 提供

        if progress_callback:
            progress_callback("searching", 0.3)

        if self._stop or not resources:
            return {"plan": plan, "resources": resources, "local_files": []}

        # 3. 下载
        self._log(f"Step 3/3: 下载 {len(resources)} 个资源...")
        local_files = []
        for i, res in enumerate(resources):
            if self._stop:
                break

            if res["type"] == "video":
                if res.get("bvid"):
                    path = self.download_bilibili(res["bvid"])
                elif source == "ytdlp":
                    path = self.download_with_ytdlp(res["url"])
                else:
                    path = self.download_video_url(res["url"])
            else:
                path = self.download_image(res["url"])

            if path:
                local_files.append(path)
                res["local_path"] = path

            if progress_callback:
                progress_callback("downloading", 0.3 + 0.7 * (i + 1) / len(resources))

        self._log(f"下载完成: {len(local_files)}/{len(resources)} 个文件")

        # 保存搜索记录
        record_path = self._output_dir / "fetch_record.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump({"plan": plan, "resources": resources,
                       "local_files": local_files}, f, ensure_ascii=False, indent=2)

        return {"plan": plan, "resources": resources, "local_files": local_files}
