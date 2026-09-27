"""Stage 1 — download audios from YouTube channels / playlists / videos.

Reads a text file of URLs (any mix of channel, playlist, single video), expands
them to individual videos, and downloads audio at the configured format / SR /
channels into input_audios/. Resumable: our own checkpoint means re-running
only fetches what's missing.

Cloud IPs (RunPod, k8s, …) are routinely hit by YouTube's "Sign in to confirm
you're not a bot" check. Without cookies, the answer is a PO-token provider
(bgutil, installed by scripts/setup_pot_provider.sh): this stage starts its
server automatically. It also paces requests and stops early with instructions
if YouTube keeps blocking the IP (README → "YouTube on RunPod").
"""

import re
import shutil
import socket
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rich.markup import escape

from .config import resolve
from .state import Checkpoint
from .ui import banner, console, err, info, note, ok, progress, warn

POT_PORT = 4416                     # bgutil server default; the plugin looks here

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_BOT_MARKERS = ("sign in to confirm", "not a bot", "cookies", "http error 403",
                "http error 429", "too many requests")


def _clean(msg) -> str:
    """Strip yt-dlp's ANSI colors and escape rich markup ('[youtube]' etc.)."""
    return escape(_ANSI.sub("", str(msg)).strip())


def _is_bot_block(msg: str) -> bool:
    m = msg.lower()
    return any(s in m for s in _BOT_MARKERS)


def _read_sources(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _video_id(url: str) -> str | None:
    """Return the video id of a single-video URL, or None for channels/playlists.

    Parsing locally means single videos need NO network call during expansion —
    that call is exactly what triggers YouTube's bot check on cloud IPs.
    """
    u = urlparse(url if "://" in url else f"https://{url}")
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    parts = [p for p in u.path.split("/") if p]
    vid = None
    if host == "youtu.be" and parts:
        vid = parts[0]
    elif host.endswith("youtube.com"):
        if parts[:1] == ["watch"]:
            vid = parse_qs(u.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v"):
            vid = parts[1]
    return vid if vid and _VIDEO_ID.match(vid) else None


def _cookie_file(dl_cfg) -> Path | None:
    """Resolve + sanity-check download.cookies_file. Returns None if unusable."""
    if not getattr(dl_cfg, "cookies_file", None):
        return None
    p = resolve(dl_cfg.cookies_file)
    if not p.exists():
        warn(f"cookies file not found: {p}")
        note("export it from a logged-in browser and upload it there (README → YouTube cookies)")
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if "HTTP Cookie File" not in first:
        warn(f"{p.name} is not in Netscape cookies.txt format — yt-dlp will reject it.")
        note("use the 'Get cookies.txt LOCALLY' extension, or yt-dlp --cookies-from-browser")
        return None
    if "youtube.com" not in text:
        warn(f"{p.name} contains no youtube.com cookies — export while on youtube.com")
        return None
    ok(f"using cookies: [accent]{p}[/accent]")
    return p


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_pot_server(dl_cfg):
    """Start the bgutil PO-token server if it isn't running. Returns the process
    we started (so run() can stop it), or None."""
    if not getattr(dl_cfg, "pot_provider", True):
        return None
    if _port_open(POT_PORT):
        ok(f"PO-token server already running on :{POT_PORT}")
        return None
    try:
        import yt_dlp_plugins.extractor.getpot_bgutil  # noqa: F401
    except ImportError:
        warn("PO-token plugin not installed — YouTube will likely block this cloud IP.")
        note("run once:  bash scripts/setup_pot_provider.sh")
        return None
    main_js = resolve(getattr(dl_cfg, "pot_server_home",
                              ".tools/bgutil-ytdlp-pot-provider/server")) / "build" / "main.js"
    if not (main_js.exists() and shutil.which("node")):
        warn(f"PO-token server not built ({main_js}) or node missing.")
        note("run once:  bash scripts/setup_pot_provider.sh")
        return None
    log_path = resolve(getattr(dl_cfg, "pot_log", ".tools/pot_server.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    proc = subprocess.Popen(["node", str(main_js)], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):                     # wait up to ~30 s for it to listen
        if _port_open(POT_PORT):
            ok(f"PO-token server started on :{POT_PORT}")
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    warn("PO-token server failed to start — see .tools/pot_server.log")
    proc.terminate()
    return None


def _preflight() -> None:
    """Warn about the two other common causes of YouTube failures."""
    # YouTube now needs a JS runtime to solve its player challenges; deno is
    # what yt-dlp enables by default.
    if not shutil.which("deno"):
        warn("deno not found on PATH — yt-dlp needs it to solve YouTube's JS challenges.")
        note("install it:  conda install -c conda-forge deno -y")
    # YouTube changes constantly; an old yt-dlp is the #1 cause of breakage.
    try:
        from yt_dlp.version import __version__ as v
        age = (date.today() - datetime.strptime(v[:10], "%Y.%m.%d").date()).days
        if age > 45:
            warn(f"yt-dlp {v} is {age} days old — YouTube breaks old versions.")
            note('update it:  pip install -U "yt-dlp[default]"')
    except Exception:
        pass


def _base_opts(dl_cfg, cookies: Path | None) -> dict:
    """Options shared by expansion and download (auth, proxy, pacing)."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,              # our rich bar is the progress display
        # pace metadata requests; bursts are what trip the bot check
        "sleep_interval_requests": getattr(dl_cfg, "sleep_requests_s", 1),
    }
    if cookies:
        opts["cookiefile"] = str(cookies)
    if getattr(dl_cfg, "proxy", None):
        opts["proxy"] = dl_cfg.proxy
    extractor_args = getattr(dl_cfg, "extractor_args", None)
    if extractor_args:
        # config gives {youtube: {player_client: "tv,web"}}; yt-dlp wants lists
        opts["extractor_args"] = {
            ie: {k: (v if isinstance(v, list) else str(v).split(","))
                 for k, v in vars(args).items()}
            for ie, args in vars(extractor_args).items()
        }
    return opts


def _expand(urls: list[str], dl_cfg, cookies: Path | None) -> list[dict]:
    """Flatten channels/playlists into individual {id, title, url} video entries."""
    import yt_dlp

    flat = []
    seen = set()

    def add(vid, title=None):
        if vid and vid not in seen:
            seen.add(vid)
            # Always download via the canonical watch URL built from the video id.
            # The flat-extraction "url" can be a resolved STREAM url (googlevideo),
            # which makes yt-dlp skip metadata and name the file after the huge url.
            flat.append({"id": vid, "title": title or vid,
                         "url": f"https://www.youtube.com/watch?v={vid}"})

    remote = []
    for url in urls:
        vid = _video_id(url)
        if vid:
            add(vid)                 # single video: no network needed
        else:
            remote.append(url)

    if not remote:
        return flat

    opts = {
        **_base_opts(dl_cfg, cookies),
        "extract_flat": "in_playlist",   # list entries without downloading
        "ignoreerrors": dl_cfg.ignore_errors,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in remote:
            note(f"resolving {url}")
            try:
                data = ydl.extract_info(url, download=False)
            except Exception as e:
                err(f"could not resolve {url}: {_clean(e)}")
                continue
            if not data:
                err(f"could not resolve {url}")
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
                add(e.get("id"), e.get("title"))
    return flat


def _ydl_opts(out_dir: Path, dl_cfg, cookies: Path | None) -> dict:
    pp = [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": dl_cfg.audio_format,
        "preferredquality": dl_cfg.mp3_bitrate.rstrip("k") if dl_cfg.audio_format == "mp3" else "0",
    }]
    # force SR + channel count via ffmpeg post-args
    postargs = ["-ar", str(dl_cfg.sample_rate), "-ac", str(dl_cfg.channels)]
    return {
        **_base_opts(dl_cfg, cookies),
        "format": "bestaudio/best",
        # name strictly by video id. (No trim_file_name: yt-dlp applies it to the
        # WHOLE path, so a long out_dir got truncated and files vanished.)
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "windowsfilenames": True,             # also forbids '?' etc. in names
        # We handle per-video errors ourselves (try/except below) so the actual
        # failure reason is surfaced instead of being silently swallowed.
        "ignoreerrors": False,
        "retries": dl_cfg.retries,
        "fragment_retries": dl_cfg.retries,
        # random pause between videos — keeps a cookie'd account from being flagged
        "sleep_interval": getattr(dl_cfg, "sleep_min_s", 3),
        "max_sleep_interval": getattr(dl_cfg, "sleep_max_s", 8),
        # NOTE: we deliberately do NOT use yt-dlp's download_archive. It marks a
        # video "done" BEFORE post-processing, so a failed ffmpeg convert poisons
        # the archive and the video is silently skipped forever. Our own Checkpoint
        # (below) only records a video once the final .mp3 actually exists on disk.
        "postprocessors": pp,
        # key is the PP name without "FFmpeg" — "ffmpegextractaudio" was silently
        # ignored, so audio kept the source SR / channels
        "postprocessor_args": {"extractaudio": postargs},
        "prefer_ffmpeg": True,
    }


def _bot_help(cookies: Path | None, pot_running: bool) -> None:
    err("YouTube is blocking this machine (\"Sign in to confirm you're not a bot\").")
    if not pot_running:
        note("the PO-token provider is not running — set it up first:")
        note("  bash scripts/setup_pot_provider.sh   then re-run")
    elif cookies:
        note("your cookies were rejected — re-export them (README → YouTube on RunPod).")
    else:
        note("this pod's IP is flagged even with PO tokens. Options (README → YouTube on RunPod):")
        note("  1. download on your home PC and send input_audios/ to the pod (runpodctl)")
        note("  2. set download.proxy to a residential proxy")
        note("  3. restart on a different pod / region (new IP) and re-run")
    note("finished videos are checkpointed — re-running only fetches what's missing.")


def run(cfg) -> None:
    dl = cfg.download
    banner("Stage 1 · Download",
           f"format={dl.audio_format}  sr={dl.sample_rate}Hz  ch={dl.channels}")

    # ffmpeg/ffprobe are required for audio extraction + conversion.
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

    _preflight()
    cookies = _cookie_file(dl)
    pot_proc = _start_pot_server(dl)
    pot_running = _port_open(POT_PORT)
    try:
        _download_all(cfg, sources, out_dir, state_dir, cookies, pot_running)
    finally:
        if pot_proc:
            pot_proc.terminate()
            try:
                pot_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pot_proc.kill()


def _download_all(cfg, sources, out_dir, state_dir, cookies, pot_running) -> None:
    dl = cfg.download

    urls = _read_sources(sources)
    info(f"{len(urls)} source line(s) in {sources.name}")

    with console.status("[info]expanding channels / playlists …[/info]", spinner="dots"):
        videos = _expand(urls, dl, cookies)
    ok(f"{len(videos)} unique video(s) to consider")

    ckpt = Checkpoint(state_dir, "download")
    import yt_dlp
    opts = _ydl_opts(out_dir, dl, cookies)
    # stop after this many back-to-back bot blocks instead of failing every video
    abort_after = getattr(dl, "bot_abort_after", 5)

    done = skipped = failed = 0
    bot_streak = 0
    aborted = False
    with progress() as bar:
        task = bar.add_task("[accent]downloading[/accent]", total=len(videos))
        with yt_dlp.YoutubeDL(opts) as ydl:
            for v in videos:
                target = out_dir / f"{v['id']}.{dl.audio_format}"
                if ckpt.done(v["id"]) or target.exists():
                    skipped += 1
                    bar.advance(task)
                    continue
                bar.update(task, description=f"[accent]⬇[/accent] {escape(v['title'][:50])}")
                try:
                    ydl.download([v["url"]])
                    if target.exists():
                        ckpt.mark(v["id"])
                        done += 1
                        bot_streak = 0
                    else:
                        failed += 1
                        warn(f"failed {v['id']}: finished but {target.name} was not created")
                except Exception as e:
                    failed += 1
                    msg = _clean(e)
                    warn(f"failed {v['id']}: {msg}")
                    if _is_bot_block(msg):
                        bot_streak += 1
                        if abort_after and bot_streak >= abort_after:
                            aborted = True
                            break
                    else:
                        bot_streak = 0
                bar.advance(task)

    ok(f"downloaded {done} • skipped {skipped} • failed {failed}")
    info(f"audios in [accent]{out_dir}[/accent]")
    if aborted:
        _bot_help(cookies, pot_running)
        # stop the full pipeline too: later stages would run on a partial download
        raise SystemExit(1)
