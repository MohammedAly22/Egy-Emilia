"""Stage 6 — publish the clean, transcribed chunks to the HuggingFace Hub as a
PRIVATE dataset.

Builds an HF "audiofolder" layout in staging_dir (hardlinks, no extra disk):

  hf_dataset/
  ├── README.md            dataset card
  ├── metadata.jsonl       file_name, text, duration, speaker, start, end, scores …
  └── audio/<video_id>/chunk_00001_spk0.mp3

then uploads it with upload_large_folder (resumable, multi-commit, safe for
tens of thousands of files). Load it later with:

  load_dataset("<repo_id>", split="train", token=True)

Auth: set HF_TOKEN in the environment, or run `hf auth login` once.
"""

import json
import os
import shutil
from pathlib import Path

from .config import resolve
from .ui import banner, err, info, note, ok, warn


def _link(src: Path, dst: Path) -> None:
    """Hardlink src -> dst (instant, no extra disk); copy if linking isn't possible."""
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _card(repo_id: str, n: int, hours: float, cfg) -> str:
    return f"""---
language:
- ar
- en
pretty_name: EGY-Emilia
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- egyptian-arabic
- code-switching
- tts
---

# EGY-Emilia

Clean, single-speaker Egyptian-Arabic speech chunks (with Arabic/English
code-switching) for TTS training, built with the EGY-Emilia pipeline.

- **Chunks:** {n}
- **Total audio:** {hours:.2f} h
- **Sample rate:** {cfg.diarization.write_sr} Hz, mono, {cfg.diarization.write_format}
- **Quality gate:** `{cfg.quality.filter_metric} >= {cfg.quality.filter_threshold}`
- **ASR backend:** `{cfg.asr.backend}`

## Columns

| column | description |
|---|---|
| `audio` | the chunk |
| `text` | transcription |
| `duration` | seconds |
| `speaker` | diarization speaker label (local to its source video) |
| `source_id` | YouTube video id the chunk was cut from |
| `start`, `end` | position in the source video (s) |
| `scores`, `overall` | quality sub-scores and blended score |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train", token=True)
```
"""


def run(cfg) -> None:
    pub = getattr(cfg, "publish", None)
    banner("Stage 6 · Publish to HuggingFace",
           "private dataset · clean + transcribed chunks")

    if pub is None or not getattr(pub, "enabled", True):
        info("publish disabled in config.yaml — skipping")
        return
    if not getattr(pub, "repo_id", None):
        warn("publish.repo_id is not set in config.yaml — skipping")
        return

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        err("huggingface_hub is not installed.")
        note("install it:  pip install -U huggingface_hub hf_xet")
        return

    out_root = resolve(cfg.paths.output)
    clean_path = out_root / "chunks_clean.json"
    if not clean_path.exists():
        warn("chunks_clean.json not found — run quality + transcribe first")
        return
    records = json.loads(clean_path.read_text(encoding="utf-8"))

    require_text = getattr(pub, "require_text", True)
    keep = []
    missing_audio = no_text = 0
    for r in records:
        rel = r["chunk_path"].replace("\\", "/")
        if not (out_root / rel).exists():
            missing_audio += 1
            continue
        if require_text and not (r.get("text") or "").strip():
            no_text += 1
            continue
        keep.append((rel, r))
    if missing_audio:
        warn(f"{missing_audio} chunk(s) listed in chunks_clean.json are missing on disk")
    if no_text:
        warn(f"{no_text} chunk(s) have no transcription — skipped (publish.require_text)")
    if not keep:
        err("nothing to publish")
        return

    # ---- stage the audiofolder layout ------------------------------------ #
    stage = resolve(getattr(pub, "staging_dir", "hf_dataset"))
    stage.mkdir(parents=True, exist_ok=True)
    wanted = set()
    rows = []
    for rel, r in keep:
        source_id, _, name = rel.partition("/chunks/")
        file_name = f"audio/{source_id}/{name}"
        wanted.add(file_name)
        _link(out_root / rel, stage / file_name)
        rows.append({
            "file_name": file_name,
            "text": (r.get("text") or "").strip(),
            "duration": r.get("duration"),
            "speaker": r.get("speaker"),
            "source_id": source_id,
            "start": r.get("start"),
            "end": r.get("end"),
            "overall": r.get("overall"),
            "scores": r.get("scores") or {},
        })
    # drop stale files from earlier runs (e.g. chunks later filtered out)
    for f in (stage / "audio").rglob("*"):
        if f.is_file() and f.relative_to(stage).as_posix() not in wanted:
            f.unlink()

    hours = sum(r["duration"] or 0 for r in rows) / 3600
    with open(stage / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (stage / "README.md").write_text(_card(pub.repo_id, len(rows), hours, cfg),
                                     encoding="utf-8")
    ok(f"staged {len(rows)} chunk(s) · {hours:.2f} h → [accent]{stage}[/accent]")

    # ---- upload ------------------------------------------------------------ #
    api = HfApi(token=getattr(pub, "token", None) or None)
    try:
        user = api.whoami()["name"]
    except Exception:
        err("not logged in to HuggingFace.")
        note("run:  hf auth login     (or:  export HF_TOKEN=hf_xxx)")
        note("the token needs WRITE access: https://huggingface.co/settings/tokens")
        return
    info(f"logged in as [accent]{user}[/accent]")

    private = getattr(pub, "private", True)
    try:
        api.create_repo(pub.repo_id, repo_type="dataset", private=private, exist_ok=True)
        # create_repo(exist_ok=True) leaves an EXISTING repo's visibility alone,
        # so enforce it explicitly.
        api.update_repo_settings(pub.repo_id, repo_type="dataset", private=private)
    except HfHubHTTPError as e:
        err(f"could not create / configure {pub.repo_id}: {e}")
        return

    info(f"uploading to [accent]{pub.repo_id}[/accent] "
         f"({'private' if private else 'PUBLIC'}) — resumable, re-run if interrupted")
    api.upload_large_folder(
        repo_id=pub.repo_id,
        folder_path=stage,
        repo_type="dataset",
        private=private,
        ignore_patterns=[".cache/**"],
        print_report=False,
    )
    ok(f"published → [accent]https://huggingface.co/datasets/{pub.repo_id}[/accent]")
    if private:
        note("private: only you (and orgs/users you grant access) can see it")
