"""Full pipeline orchestrator:
download -> diarize -> loudness -> quality -> transcribe -> publish.

Each stage is independently resumable, so running the whole thing again only does
the work that's left.
"""

import time

from . import diarize_stage, download, loudness, publish, quality, transcribe
from .config import load_config
from .ui import banner, console, ok


STAGES = [
    ("download", download.run),
    ("diarize", diarize_stage.run),
    ("loudness", loudness.run),
    ("quality", quality.run),
    ("transcribe", transcribe.run),
    ("publish", publish.run),
]


def run_all(cfg=None) -> None:
    cfg = cfg or load_config()
    banner("EGY-Emilia · Full Pipeline",
           "TTS data collection · clean single-speaker chunks")
    t0 = time.time()
    for name, fn in STAGES:
        fn(cfg)
    console.rule("[ok]pipeline complete[/ok]")
    ok(f"all stages done in {(time.time()-t0)/60:.1f} min")
