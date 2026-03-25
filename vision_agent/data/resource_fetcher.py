"""网络资源搜索与下载：用 LLM 规划搜索策略，自动获取视频/图片。"""

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class ResourceFetcher:
    """LLM 驱动的网络资源搜索与下载。

    流程: 用户兴趣描述 → LLM 规划搜索关键词 → 搜索引擎获取资源 → 下载到本地
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

    def plan_search(self, interest: str, resource_type: str = "video") -> dict:
        """让 LLM 根据用户兴趣规划搜索策略。

        Returns:
            {"keywords": [...], "description": "...", "suggested_actions": [...]}
        """
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
- keywords: 3-5个中文搜索关键词，用于搜索相关视频/图片
- keywords_en: 对应的英文关键词（搜索范围更广）
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
                # 提取 JSON
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

        # Fallback: 基础策略
        return {
            "keywords": [interest],
            "keywords_en": [interest],
            "description": interest,
            "suggested_actions": ["attack", "defend", "move", "skill", "idle"],
            "action_descriptions": {},
        }

    def search_videos_yt(self, keywords: list[str], max_results: int = 5) -> list[dict]:
        """用 yt-dlp 搜索 YouTube 视频（不下载，只获取信息）。"""
        results = []
        try:
            import yt_dlp
        except ImportError:
            self._log("[提示] 需要安装 yt-dlp: pip install yt-dlp")
            return results

        for kw in keywords:
            if self._stop:
                break
            self._log(f"搜索视频: {kw}")
            opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": "ytsearch",
            }
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(f"ytsearch{max_results}:{kw}", download=False)
                    for entry in (info.get("entries") or []):
                        if entry and entry.get("url"):
                            results.append({
                                "type": "video",
                                "url": entry["url"],
                                "title": entry.get("title", ""),
                                "duration": entry.get("duration", 0),
                                "source": "youtube",
                                "keyword": kw,
                            })
            except Exception as e:
                self._log(f"搜索失败 ({kw}): {e}")

        self._log(f"共找到 {len(results)} 个视频")
        return results

    def search_images(self, keywords: list[str], max_per_keyword: int = 10) -> list[dict]:
        """搜索图片 URL（通过公开 API）。"""
        results = []
        for kw in keywords:
            if self._stop:
                break
            self._log(f"搜索图片: {kw}")
            try:
                # 使用 Unsplash 免费 API（无需 key 的搜索）
                resp = requests.get(
                    "https://api.unsplash.com/search/photos",
                    params={"query": kw, "per_page": max_per_keyword},
                    headers={"Accept-Version": "v1"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    for item in resp.json().get("results", []):
                        url = item.get("urls", {}).get("regular", "")
                        if url:
                            results.append({
                                "type": "image",
                                "url": url,
                                "title": item.get("alt_description", ""),
                                "source": "unsplash",
                                "keyword": kw,
                            })
            except Exception as e:
                self._log(f"图片搜索失败 ({kw}): {e}")

        self._log(f"共找到 {len(results)} 张图片")
        return results

    def download_video(self, url: str, filename: str = "") -> str | None:
        """下载视频到本地。"""
        try:
            import yt_dlp
        except ImportError:
            self._log("[错误] 需要安装 yt-dlp")
            return None

        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "worst[ext=mp4]",  # 下载最小分辨率节省空间
            "outtmpl": str(out_dir / (filename or "%(title)s.%(ext)s")),
            "max_filesize": 100 * 1024 * 1024,  # 100MB 上限
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = ydl.prepare_filename(info)
                self._log(f"下载完成: {Path(path).name}")
                return path
        except Exception as e:
            self._log(f"下载失败: {e}")
            return None

    def download_image(self, url: str, filename: str = "") -> str | None:
        """下载图片到本地。"""
        out_dir = self._output_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = f"img_{int(time.time()*1000)}.jpg"
        path = out_dir / filename

        try:
            resp = requests.get(url, timeout=15, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return str(path)
        except Exception as e:
            self._log(f"图片下载失败: {e}")
            return None

    def fetch(self, interest: str, resource_type: str = "video",
              max_results: int = 5, progress_callback=None) -> dict:
        """完整的搜索+下载流程。

        Returns:
            {"plan": {...}, "resources": [...], "local_files": [...]}
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
        all_keywords = plan.get("keywords", []) + plan.get("keywords_en", [])

        if resource_type == "video":
            resources = self.search_videos_yt(all_keywords, max_results=max_results)
        else:
            resources = self.search_images(all_keywords, max_per_keyword=max_results)

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
                path = self.download_video(res["url"])
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
