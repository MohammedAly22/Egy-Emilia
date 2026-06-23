"""Stage 3 — EBU R128 loudness normalization (in place) on every chunk.

Uses ffmpeg's loudnorm filter. Idempotent + resumable via a per-chunk checkpoint.
"""

import subprocess
from pathlib import Path

from .config import resolve
from .state import Checkpoint
from .ui import banner, info, ok, progress, warn


def _loudnorm(src: Path, lo) -> None:
    """Normalize src in place to target LUFS / true-peak / LRA."""
    af = f"loudnorm=I={lo.target_lufs}:TP={lo.true_peak}:LRA={lo.lra}"
    # Write the temp output IN THE SAME DIRECTORY as src so the final atomic
    # replace() is a same-filesystem rename. Using /tmp fails with
    # "Invalid cross-device link" when /tmp is a different mount.
    tmp_path = src.with_name(f".{src.stem}.norm{src.suffix}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-af", af, str(tmp_path)],
            check=True,
        )
        tmp_path.replace(src)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run(cfg) -> None:
    lo = cfg.loudness
    banner("Stage 3 · Loudness Normalization",
           f"target {lo.target_lufs} LUFS · TP {lo.true_peak} dBTP")
    if not lo.enabled:
        info("loudness normalization disabled in config — skipping")
        return

    out_root = resolve(cfg.paths.output)
    state_dir = resolve(cfg.paths.state_dir)
    ckpt = Checkpoint(state_dir, "loudness")

    chunks = sorted(out_root.glob("*/chunks/*"))
    chunks = [c for c in chunks if c.suffix in (".wav", ".mp3")]
    if not chunks:
        warn(f"no chunks found under {out_root} — run diarization first")
        return

    done = skipped = 0
    with progress() as bar:
        task = bar.add_task("[accent]normalizing[/accent]", total=len(chunks))
        for c in chunks:
            key = str(c.relative_to(out_root))
            if ckpt.done(key):
                skipped += 1
                bar.advance(task)
                continue
            try:
                _loudnorm(c, lo)
                ckpt.mark(key)
                done += 1
            except Exception as e:
                warn(f"loudnorm failed {key}: {e}")
            bar.advance(task)
    ok(f"normalized {done} • skipped {skipped}")
