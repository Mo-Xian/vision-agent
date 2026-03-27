"""视频下载器：从在线平台下载游戏视频。

依赖 yt-dlp（可选）。未安装时跳过下载并提示安装。
支持 Bilibili、YouTube 等主流平台。
"""

import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ytdlp_available() -> bool:
    """检查 yt-dlp 是否已安装。"""
    return shutil.which("yt-dlp") is not None


def download_video(
    url: str,
    output_dir: str,
    max_duration: int = 600,
) -> str | None:
    """下载单个视频。

    Args:
        url: 视频 URL（支持 Bilibili/YouTube 等）
        output_dir: 输出目录
        max_duration: 最大时长（秒），超过则跳过

    Returns:
        下载的视频文件路径，失败返回 None
    """
    if not is_ytdlp_available():
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    template = str(out / "%(title).50s.%(ext)s")

    try:
        cmd = [
            "yt-dlp",
            "--match-filter", f"duration<{max_duration}",
            "-f", "best[height<=720]/best",
            "--no-playlist",
            "--no-overwrites",
            "-o", template,
            "--print", "after_move:filepath",
            url,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0:
            filepath = proc.stdout.strip().split("\n")[-1]
            if Path(filepath).exists():
                return filepath
        logger.warning(f"下载失败 [{url[:60]}]: {proc.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"下载超时: {url[:60]}")
    except Exception as e:
        logger.warning(f"下载异常: {e}")

    return None


def download_videos(
    urls: list[str],
    output_dir: str,
    max_count: int = 5,
    max_duration: int = 600,
    on_log=None,
) -> list[str]:
    """批量下载视频。

    Args:
        urls: 视频 URL 列表
        output_dir: 输出目录
        max_count: 最多下载几个
        max_duration: 单个视频最大时长（秒）
        on_log: 日志回调

    Returns:
        成功下载的视频文件路径列表
    """
    def _log(msg):
        logger.info(msg)
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    if not is_ytdlp_available():
        _log("[下载] yt-dlp 未安装，无法自动下载视频。请运行: pip install yt-dlp")
        return []

    valid_urls = [u for u in urls if u and u.startswith("http")]
    if not valid_urls:
        _log("[下载] 无有效的视频 URL")
        return []

    _log(f"[下载] 准备下载 {min(len(valid_urls), max_count)} 个视频...")
    downloaded = []

    for i, url in enumerate(valid_urls[:max_count]):
        if len(downloaded) >= max_count:
            break
        _log(f"  [{i+1}/{min(len(valid_urls), max_count)}] {url[:80]}...")
        path = download_video(url, output_dir, max_duration)
        if path:
            _log(f"    -> {Path(path).name}")
            downloaded.append(path)
        else:
            _log(f"    -> 跳过（下载失败或视频过长）")

    _log(f"[下载] 完成: {len(downloaded)}/{min(len(valid_urls), max_count)} 个视频")
    return downloaded
