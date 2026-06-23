"""Stage 1 — download audios from YouTube channels / playlists / videos.

Reads a text file of URLs (any mix of channel, playlist, single video), expands
them to individual videos, and downloads audio at the configured format / SR /
channels into input_audios/. Resumable: yt-dlp's download-archive plus our own
checkpoint mean re-running only fetches what's missing.
"""

from pathlib import Path

from .config import resolve
from .state import Checkpoint
from .ui import banner, console, err, info, note, ok, progress, warn


def _read_sources(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _expand(urls: list[str], dl_cfg) -> list[dict]:
    """Flatten channels/playlists into individual {id, title, url} video entries."""
    import yt_dlp

    flat = []
    seen = set()
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",   # list entries without downloading
        "ignoreerrors": dl_cfg.ignore_errors,
    }
    if dl_cfg.cookies_file:
        opts["cookiefile"] = dl_cfg.cookies_file

    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            note(f"resolving {url}")
            try:
                data = ydl.extract_info(url, download=False)
            except Exception as e:
                err(f"could not resolve {url}: {e}")
                continue
            entries = data.get("entries") if isinstance(data, dict) else None
            if entries is None:                      # single video
                entries = [data]
            stack = list(entries)
            while stack:                             # channels -> tabs -> playlists
                e = stack.pop()
                if e is None:
                    continue
                if e.get("entries"):
                    stack.extend(e["entries"])
                    continue
                vid = e.get("id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                # Always download via the canonical watch URL built from the video id.
                # The flat-extraction "url" can be a resolved STREAM url (googlevideo),
                # which makes yt-dlp skip metadata and name the file after the huge url.
                flat.append({
                    "id": vid,
                    "title": e.get("title") or vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                })
    return flat


def _ydl_opts(out_dir: Path, dl_cfg) -> dict:
    pp = [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": dl_cfg.audio_format,
        "preferredquality": dl_cfg.mp3_bitrate.rstrip("k") if dl_cfg.audio_format == "mp3" else "0",
    }]
    # force SR + channel count via ffmpeg post-args
    postargs = ["-ar", str(dl_cfg.sample_rate), "-ac", str(dl_cfg.channels)]
    opts = {
        "format": "bestaudio/best",
        # name strictly by video id; trim_file_name guards against any long fallback
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "trim_file_name": 100,
        "restrictfilenames": True,
        "windowsfilenames": True,             # also forbids '?' etc. in names
        "quiet": True,
        "no_warnings": True,
        # We handle per-video errors ourselves (try/except below) so the actual
        # failure reason is surfaced instead of being silently swallowed.
        "ignoreerrors": False,
        "retries": dl_cfg.retries,
        # NOTE: we deliberately do NOT use yt-dlp's download_archive. It marks a
        # video "done" BEFORE post-processing, so a failed ffmpeg convert poisons
        # the archive and the video is silently skipped forever. Our own Checkpoint
        # (below) only records a video once the final .mp3 actually exists on disk.
        "postprocessors": pp,
        "postprocessor_args": {"ffmpegextractaudio": postargs},
        "prefer_ffmpeg": True,
    }
    if dl_cfg.cookies_file:
        opts["cookiefile"] = dl_cfg.cookies_file
    return opts


def run(cfg) -> None:
    dl = cfg.download
    banner("Stage 1 · Download",
           f"format={dl.audio_format}  sr={dl.sample_rate}Hz  ch={dl.channels}")

    # ffmpeg/ffprobe are required for audio extraction + conversion.
    import shutil
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        err("ffmpeg/ffprobe not found on PATH.")
        note("install it:  conda install -c conda-forge ffmpeg -y")
        return

    sources = resolve(cfg.paths.sources_file)
    out_dir = resolve(cfg.paths.input_audios)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = resolve(cfg.paths.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    if not sources.exists():
        err(f"sources file not found: {sources}")
        return

    urls = _read_sources(sources)
    info(f"{len(urls)} source line(s) in {sources.name}")

    with console.status("[info]expanding channels / playlists …[/info]", spinner="dots"):
        videos = _expand(urls, dl)
    ok(f"{len(videos)} unique video(s) to consider")

    ckpt = Checkpoint(state_dir, "download")
    import yt_dlp
    opts = _ydl_opts(out_dir, dl)

    done = skipped = failed = 0
    last_err = ""
    with progress() as bar:
        task = bar.add_task("[accent]downloading[/accent]", total=len(videos))
        with yt_dlp.YoutubeDL(opts) as ydl:
            for v in videos:
                target = out_dir / f"{v['id']}.{dl.audio_format}"
                if ckpt.done(v["id"]) or target.exists():
                    skipped += 1
                    bar.advance(task)
                    continue
                bar.update(task, description=f"[accent]⬇[/accent] {v['title'][:50]}")
                try:
                    ydl.download([v["url"]])
                    if target.exists():
                        ckpt.mark(v["id"])
                        done += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    last_err = str(e)
                    warn(f"failed {v['id']}: {e}")
                bar.advance(task)

    ok(f"downloaded {done} • skipped {skipped} • failed {failed}")
    info(f"audios in [accent]{out_dir}[/accent]")

    # Cloud/datacenter IPs (k8s pods) frequently hit YouTube bot-detection.
    if failed and done == 0 and any(
        s in last_err.lower() for s in ("sign in", "bot", "confirm", "cookies", "403")
    ):
        warn("YouTube is blocking this IP (bot check). This is common on cloud pods.")
        note("Fix: export cookies from a logged-in browser to cookies.txt, then set")
        note("download.cookies_file: \"cookies.txt\" in config.yaml and re-run.")
