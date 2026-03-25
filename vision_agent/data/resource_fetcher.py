"""网络资源搜索与下载：用 LLM 规划搜索策略，自动获取视频/图片。

支持的视频源:
  - Bilibili（B站）搜索 + yt-dlp 下载（最可靠）
  - Bilibili API 直接下载（回退方案）
  - 直接 URL 下载（任意视频链接）
  - yt-dlp 支持的所有站点
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
}

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v"}


def _check_ytdlp() -> bool:
    """检查 yt-dlp 是否可用。"""
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def _install_ytdlp() -> bool:
    """尝试自动安装 yt-dlp。"""
    try:
        subprocess.check_call(
            ["pip", "install", "yt-dlp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return True
    except Exception:
        return False


class ResourceFetcher:
    """LLM 驱动的网络资源搜索与下载。

    流程: 用户兴趣描述 → LLM 规划搜索关键词 → 搜索获取资源 → 下载到本地

    B站下载策略（按优先级）:
      1. yt-dlp 下载（最可靠，处理防盗链/登录/合并音视频）
      2. Bilibili API 直接下载（回退，可能被 403）
    """

    def __init__(self, llm_provider=None, output_dir: str = "data/fetched",
                 on_log=None, bilibili_cookie: str = ""):
        self._llm = llm_provider
        self._output_dir = Path(output_dir)
        self._on_log = on_log
        self._stop = False
        self._bilibili_cookie = bilibili_cookie or os.environ.get("BILIBILI_COOKIE", "")

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
        headers = {**_HEADERS}
        if self._bilibili_cookie:
            headers["Cookie"] = self._bilibili_cookie

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
                    headers=headers,
                    timeout=10,
                )
                data = resp.json()
                for item in (data.get("data", {}).get("result", []) or []):
                    bvid = item.get("bvid", "")
                    if not bvid:
                        continue
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

    # ── yt-dlp B站搜索（备用） ──

    def search_bilibili_ytdlp(self, keywords: list[str], max_results: int = 5) -> list[dict]:
        """通过 yt-dlp 的 bilisearch 搜索B站视频（不依赖 API）。"""
        if not _check_ytdlp():
            self._log("[提示] yt-dlp 未安装，回退到 API 搜索")
            return self.search_bilibili(keywords, max_results)

        import yt_dlp

        results = []
        for kw in keywords:
            if self._stop:
                break
            self._log(f"yt-dlp B站搜索: {kw}")
            try:
                opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "extract_flat": True,
                    "playlist_items": f"1:{max_results}",
                }
                if self._bilibili_cookie:
                    opts["http_headers"] = {"Cookie": self._bilibili_cookie}

                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(f"bilisearch{max_results}:{kw}", download=False)
                    for entry in (info or {}).get("entries", []) or []:
                        url = entry.get("url") or entry.get("webpage_url", "")
                        title = entry.get("title", kw)
                        bvid = ""
                        # 从 URL 提取 bvid
                        m = re.search(r"/(BV[\w]+)", url)
                        if m:
                            bvid = m.group(1)
                        results.append({
                            "type": "video",
                            "url": url,
                            "bvid": bvid,
                            "title": title,
                            "duration": entry.get("duration", ""),
                            "source": "bilibili",
                            "keyword": kw,
                        })
            except Exception as e:
                self._log(f"yt-dlp 搜索失败 ({kw}): {e}")

        # 去重
        seen = set()
        unique = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)

        self._log(f"yt-dlp 搜索共找到 {len(unique)} 个视频")
        return unique

    # ── B站下载（yt-dlp 优先，API 回退） ──

    def download_bilibili(self, bvid: str, url: str = "", filename: str = "") -> str | None:
        """下载 B站视频。优先用 yt-dlp，失败则回退到 API 直接下载。"""
        video_url = url or f"https://www.bilibili.com/video/{bvid}"

        # 策略1: yt-dlp（最可靠）
        path = self._download_bilibili_ytdlp(video_url, bvid, filename)
        if path:
            return path

        # 策略2: API 直接下载（可能被 403）
        self._log(f"yt-dlp 下载失败，尝试 API 直接下载: {bvid}")
        return self._download_bilibili_api(bvid, filename)

    def _download_bilibili_ytdlp(self, url: str, bvid: str, filename: str = "") -> str | None:
        """用 yt-dlp 下载B站视频。"""
        if not _check_ytdlp():
            self._log("[提示] yt-dlp 未安装，尝试自动安装...")
            if _install_ytdlp():
                self._log("[成功] yt-dlp 安装完成")
            else:
                self._log("[失败] yt-dlp 安装失败，请手动执行: pip install yt-dlp")
                return None

        import yt_dlp

        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = f"{bvid}.mp4"
        out_path = str(out_dir / filename)

        # 去掉扩展名，yt-dlp 会自动加
        out_template = out_path.rsplit(".", 1)[0] + ".%(ext)s"

        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "max_filesize": 200 * 1024 * 1024,  # 200MB
            "socket_timeout": 30,
            "retries": 3,
            "merge_output_format": "mp4",
        }

        # B站 cookie（提高下载成功率和画质）
        if self._bilibili_cookie:
            opts["http_headers"] = {"Cookie": self._bilibili_cookie}

        # 检查 cookies 文件
        cookie_file = Path("bilibili_cookies.txt")
        if cookie_file.exists():
            opts["cookiefile"] = str(cookie_file)
            self._log("使用 bilibili_cookies.txt")

        self._log(f"yt-dlp 下载B站视频: {bvid}")

        try:
            downloaded_path = None

            def _progress_hook(d):
                nonlocal downloaded_path
                if d["status"] == "downloading":
                    pct = d.get("_percent_str", "?")
                    speed = d.get("_speed_str", "?")
                    self._log(f"  下载: {pct} | 速度: {speed}")
                elif d["status"] == "finished":
                    downloaded_path = d.get("filename", "")
                    self._log(f"  下载完成，处理中...")

            opts["progress_hooks"] = [_progress_hook]

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                final_path = downloaded_path or ydl.prepare_filename(info)

                # yt-dlp 可能输出非 mp4，检查实际文件
                final = Path(final_path)
                if not final.exists():
                    # 尝试 .mp4 扩展名
                    mp4_path = final.with_suffix(".mp4")
                    if mp4_path.exists():
                        final = mp4_path

                if final.exists():
                    size_mb = final.stat().st_size / (1024 * 1024)
                    self._log(f"  B站视频已保存: {final.name} ({size_mb:.1f}MB)")
                    return str(final)

            self._log(f"yt-dlp 下载后未找到文件")
            return None

        except Exception as e:
            self._log(f"yt-dlp B站下载失败: {e}")
            return None

    def _download_bilibili_api(self, bvid: str, filename: str = "") -> str | None:
        """通过 Bilibili API 直接下载视频（回退方案）。"""
        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        headers = {**_HEADERS}
        if self._bilibili_cookie:
            headers["Cookie"] = self._bilibili_cookie

        try:
            # 获取 cid
            resp = requests.get(
                "https://api.bilibili.com/x/player/pagelist",
                params={"bvid": bvid},
                headers=headers,
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
                headers=headers,
                timeout=10,
            )
            play_data = resp.json().get("data", {})
            durl = play_data.get("durl", [])
            if not durl:
                self._log(f"获取视频流失败: {bvid}")
                return None

            video_url = durl[0]["url"]

            if not filename:
                filename = f"{bvid}.mp4"
            path = out_dir / filename

            self._log(f"API直接下载: {bvid}")
            resp = requests.get(
                video_url,
                headers={**headers, "Referer": f"https://www.bilibili.com/video/{bvid}"},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (1024 * 512) < 1024 * 64:
                        pct = downloaded / total * 100
                        self._log(f"  下载进度: {pct:.0f}%")

            size_kb = downloaded // 1024
            if size_kb < 10:
                self._log(f"API下载文件过小 ({size_kb}KB)，可能被拦截")
                path.unlink(missing_ok=True)
                return None

            self._log(f"API下载完成: {filename} ({size_kb}KB)")
            return str(path)

        except Exception as e:
            self._log(f"API下载失败 ({bvid}): {e}")
            return None

    # ── 通用 URL 下载 ──

    def download_video_url(self, url: str, filename: str = "") -> str | None:
        """直接下载视频 URL（支持任意 HTTP 视频链接）。"""
        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
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

    # ── yt-dlp 通用下载 ──

    def download_with_ytdlp(self, url: str, filename: str = "") -> str | None:
        """用 yt-dlp 下载视频（支持 B站/YouTube 等多平台）。"""
        if not _check_ytdlp():
            self._log("[提示] yt-dlp 未安装，尝试自动安装...")
            if not _install_ytdlp():
                self._log("[失败] yt-dlp 安装失败，尝试直接下载")
                return self.download_video_url(url, filename)

        import yt_dlp

        out_dir = self._output_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
            "outtmpl": str(out_dir / (filename or "%(title)s.%(ext)s")),
            "max_filesize": 200 * 1024 * 1024,
            "socket_timeout": 30,
            "merge_output_format": "mp4",
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

    # ── 环境检查 ──

    def check_environment(self) -> dict:
        """检查下载环境，返回各组件状态。"""
        status = {
            "yt_dlp_installed": _check_ytdlp(),
            "bilibili_cookie": bool(self._bilibili_cookie),
            "cookie_file_exists": Path("bilibili_cookies.txt").exists(),
        }

        if status["yt_dlp_installed"]:
            try:
                import yt_dlp
                status["yt_dlp_version"] = yt_dlp.version.__version__
            except Exception:
                status["yt_dlp_version"] = "unknown"

        recommendations = []
        if not status["yt_dlp_installed"]:
            recommendations.append("建议安装 yt-dlp: pip install yt-dlp")
        if not status["bilibili_cookie"] and not status["cookie_file_exists"]:
            recommendations.append(
                "建议设置B站Cookie以提高下载成功率:\n"
                "  方式1: 设置环境变量 BILIBILI_COOKIE\n"
                "  方式2: 在项目根目录放置 bilibili_cookies.txt (Netscape格式)"
            )
        status["recommendations"] = recommendations
        return status

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

        # 0. 环境检查
        env = self.check_environment()
        if env["recommendations"]:
            for rec in env["recommendations"]:
                self._log(f"[环境提示] {rec}")

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
            # 优先用 API 搜索（快），失败则用 yt-dlp 搜索
            resources = self.search_bilibili(keywords, max_results=max_results)
            if not resources and _check_ytdlp():
                self._log("API 搜索无结果，尝试 yt-dlp 搜索...")
                resources = self.search_bilibili_ytdlp(keywords, max_results=max_results)
        else:
            resources = []

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
                if res.get("source") == "bilibili" and res.get("bvid"):
                    path = self.download_bilibili(res["bvid"], res.get("url", ""))
                elif source == "ytdlp":
                    path = self.download_with_ytdlp(res["url"])
                else:
                    path = self.download_video_url(res["url"])
            else:
                path = self.download_image(res["url"])

            if path:
                local_files.append(path)
                res["local_path"] = path
                self._log(f"  [{i+1}/{len(resources)}] 成功")
            else:
                self._log(f"  [{i+1}/{len(resources)}] 失败，跳过")

            if progress_callback:
                progress_callback("downloading", 0.3 + 0.7 * (i + 1) / len(resources))

        self._log(f"下载完成: {len(local_files)}/{len(resources)} 个文件")

        # 保存搜索记录
        record_path = self._output_dir / "fetch_record.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump({"plan": plan, "resources": resources,
                       "local_files": local_files}, f, ensure_ascii=False, indent=2)

        return {"plan": plan, "resources": resources, "local_files": local_files}
