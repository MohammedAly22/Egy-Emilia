"""Stage 2 — run the diarize+chunk pipeline on every downloaded audio.

Each input audio gets its own subfolder under output/<audio_id>/ containing
chunks/ + manifest.csv + manifest.jsonl.

Resume rule (as requested): an audio is SKIPPED only if its chunks folder already
exists AND contains chunk files. If the chunks folder is missing or empty, the
audio is (re)processed — no separate checkpoint that can get out of sync.
"""

from pathlib import Path

from . import diarize_chunk
from .config import resolve
from .ui import banner, console, info, ok, warn

_AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus")


def _has_chunks(out_dir: Path) -> bool:
    cdir = out_dir / "chunks"
    return cdir.is_dir() and any(cdir.iterdir())


def run(cfg) -> None:
    banner("Stage 2 · Diarization + Chunking",
           "single-speaker · 3–30s · no overlap · clean cuts")

    in_dir = resolve(cfg.paths.input_audios)
    out_root = resolve(cfg.paths.output)
    out_root.mkdir(parents=True, exist_ok=True)

    audios = sorted(p for p in in_dir.glob("*") if p.suffix.lower() in _AUDIO_EXT)
    if not audios:
        warn(f"no audios in {in_dir} — run download first")
        return
    info(f"{len(audios)} audio(s) found")

    diar_dict = vars(cfg.diarization)
    done = processed = failed = 0
    for i, audio in enumerate(audios, 1):
        aid = audio.stem
        out_dir = out_root / aid
        if _has_chunks(out_dir):
            n = len(list((out_dir / "chunks").iterdir()))
            info(f"[{i}/{len(audios)}] {aid} — {n} chunks exist, skipping")
            done += 1
            continue
        console.rule(f"[accent]{aid}[/accent]  ({i}/{len(audios)})")
        dcfg = diarize_chunk.Config.from_dict(str(audio), str(out_dir), diar_dict)
        try:
            diarize_chunk.run(dcfg)
            if _has_chunks(out_dir):
                processed += 1
            else:
                warn(f"{aid} produced 0 chunks (check diarization output above)")
                failed += 1
        except Exception as e:
            warn(f"diarization failed for {aid}: {e}")
            failed += 1

    ok(f"diarization: {processed} processed • {done} already had chunks • {failed} produced nothing")
    info(f"output → {out_root}")
