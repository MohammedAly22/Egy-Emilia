"""Which YouTube player clients still work from THIS machine's IP?

When download fails with "Sign in to confirm you're not a bot", YouTube often
blocks some player clients but not others. This tries each one on a test video
(with the PO-token server running, like the real download stage) and prints the
config.yaml line to use for the first client that works.

  python scripts/diagnose_youtube.py [VIDEO_URL]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yt_dlp  # noqa: E402
from yt_dlp.networking import Request  # noqa: E402

from egy_emilia.config import load_config  # noqa: E402
from egy_emilia.download import (  # noqa: E402
    POT_PORT, _base_opts, _clean, _cookie_file, _port_open, _start_pot_server)
from egy_emilia.ui import console, err, info, ok, warn  # noqa: E402

CLIENTS = ["default", "tv", "tv_simply", "web_embedded", "mweb", "web_safari",
           "web", "android_vr", "visionos", "ios", "android"]
URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=ng6k7oBbvog"


def _try(client: str, dl_cfg, cookies) -> tuple[bool, str]:
    # same auth / JS-runtime / proxy settings as the real download stage
    opts = {**_base_opts(dl_cfg, cookies), "format": "bestaudio/best"}
    opts.pop("extractor_args", None)
    if client != "default":
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            meta = ydl.extract_info(URL, download=False)
        except Exception as e:
            return False, _clean(e).splitlines()[0][:150]
        # extraction can succeed while the media URL itself is 403'd — fetch 64 KB
        fmt = meta.get("requested_formats", [meta])[0]
        if not fmt.get("url"):
            return False, "no downloadable format"
        try:
            req = Request(fmt["url"], headers={**(fmt.get("http_headers") or {}),
                                               "Range": "bytes=0-65535"})
            with ydl.urlopen(req) as r:
                r.read()
        except Exception as e:
            return False, f"metadata ok, media blocked: {_clean(e)[:120]}"
        return True, f"format {fmt.get('format_id')} ({fmt.get('ext')})"


def main() -> None:
    cfg = load_config()
    dl = cfg.download
    if not getattr(dl, "pot_provider", False):
        os.environ["YTDLP_NO_PLUGINS"] = "1"       # plain yt-dlp, like the pipeline
    info(f"js_runtime={getattr(dl, 'js_runtime', False)}  "
         f"pot_provider={getattr(dl, 'pot_provider', False)}  (from config.yaml)")
    cookies = _cookie_file(dl)
    proc = _start_pot_server(dl)
    if getattr(dl, "pot_provider", False) and not _port_open(POT_PORT):
        warn("PO-token server not running — results reflect NO PO tokens")
    info(f"test video: {URL}")
    working = []
    try:
        for c in CLIENTS:
            with console.status(f"trying {c} …"):
                good, why = _try(c, dl, cookies)
            (ok if good else err)(f"{c:<13} {why}")
            if good:
                working.append(c)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)

    console.rule()
    if not working:
        err("no client works from this IP — use a fallback (README → YouTube on RunPod)")
        return
    if working[0] == "default":
        ok("the default clients work — no config change needed")
        return
    ok(f"working clients: {', '.join(working)}")
    info("put this in config.yaml → download:")
    console.print(f'  extractor_args: {{youtube: {{player_client: "{",".join(working)}"}}}}',
                  markup=False, highlight=False)


if __name__ == "__main__":
    main()
