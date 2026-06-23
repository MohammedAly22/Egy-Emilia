"""Stage 4 — per-chunk quality scoring -> chunks.json / chunks_clean.json.

Scorers are pluggable and degrade gracefully: if a scorer's model/dependency
isn't installed, that sub-score is recorded as null and the run continues.

Sub-scores:
  • UTMOS        — naturalness MOS (speechmos / torchhub)
  • DNSMOS P.835 — OVRL / SIG / BAK (torchaudio SQUIM or onnx DNSMOS)
  • cleanliness  — 1 - P(music/noise/SFX), from an audio tagger (PANNs/AST)

The single 'overall' score is a weighted blend (weights from config), then chunks
with overall >= filter_threshold are copied into chunks_clean.json.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import resolve
from .state import Checkpoint
from .ui import banner, info, note, ok, progress, warn


# --------------------------------------------------------------------------- #
# Scorers — each returns a dict of sub-scores or {} if unavailable.
# --------------------------------------------------------------------------- #
class UTMOSScorer:
    name = "utmos"

    def __init__(self, device: str):
        self.ok = False
        try:
            import torch
            # speechmos bundles UTMOS22; falls back silently if missing
            self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            self.model.eval().to(device)
            self.device = device
            self.ok = True
        except Exception as e:
            warn(f"UTMOS unavailable ({e}); skipping this sub-score")

    def score(self, wav: np.ndarray, sr: int) -> dict:
        if not self.ok:
            return {}
        import torch
        with torch.no_grad():
            t = torch.from_numpy(wav).unsqueeze(0).to(self.device)
            val = float(self.model(t, sr))
        return {"utmos": round(val, 3)}


class DNSMOSScorer:
    name = "dnsmos"

    def __init__(self, device: str):
        self.ok = False
        try:
            import torch
            import torchaudio
            self.obj = torchaudio.pipelines.SQUIM_SUBJECTIVE  # gives MOS-like
            # SQUIM_OBJECTIVE returns STOI/PESQ/SI-SDR; SUBJECTIVE returns MOS.
            self.model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device)
            self.device = device
            self.ok = True
        except Exception as e:
            warn(f"DNSMOS/SQUIM unavailable ({e}); skipping this sub-score")

    def score(self, wav: np.ndarray, sr: int) -> dict:
        if not self.ok:
            return {}
        import torch
        import torchaudio
        if sr != 16000:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, 16000).numpy()
        with torch.no_grad():
            t = torch.from_numpy(wav).unsqueeze(0).to(self.device)
            stoi, pesq, sisdr = self.model(t)
        # map PESQ (1..4.5) to an OVRL-like proxy; keep raw values too
        pesq = float(pesq)
        return {
            "dnsmos_ovrl": round(pesq, 3),     # proxy overall MOS
            "squim_stoi": round(float(stoi), 3),
            "squim_sisdr": round(float(sisdr), 3),
        }


class MusicNoiseScorer:
    name = "music_noise"

    def __init__(self, device: str):
        self.ok = False
        try:
            from transformers import pipeline
            self.clf = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=0 if device == "cuda" else -1,
            )
            self.ok = True
        except Exception as e:
            warn(f"music/noise tagger unavailable ({e}); skipping this sub-score")

    _BAD = ("music", "noise", "sound effect", "sfx", "vehicle", "instrument",
            "singing", "musical")

    def score(self, wav: np.ndarray, sr: int) -> dict:
        if not self.ok:
            return {}
        preds = self.clf({"array": wav.astype("float32"), "sampling_rate": sr}, top_k=10)
        bad = sum(p["score"] for p in preds
                  if any(b in p["label"].lower() for b in self._BAD))
        bad = min(1.0, float(bad))
        return {
            "music_noise_prob": round(bad, 3),
            "cleanliness": round(1.0 - bad, 3),
        }


# --------------------------------------------------------------------------- #
def _overall(scores: dict, weights) -> float | None:
    """Weighted blend of whichever sub-scores are present (MOS-scale ~1..5)."""
    w = vars(weights)
    terms, wsum = 0.0, 0.0
    # cleanliness is 0..1 -> scale to ~1..5 so it blends with MOS scores
    mapping = {
        "utmos": scores.get("utmos"),
        "dnsmos_ovrl": scores.get("dnsmos_ovrl"),
        "cleanliness": None if scores.get("cleanliness") is None
                       else 1.0 + 4.0 * scores["cleanliness"],
    }
    for k, val in mapping.items():
        if val is not None and w.get(k):
            terms += w[k] * val
            wsum += w[k]
    return round(terms / wsum, 3) if wsum else None


def run(cfg) -> None:
    q = cfg.quality
    banner("Stage 4 · Quality Scoring",
           f"DNSMOS={q.dnsmos}  UTMOS={q.utmos}  music/noise={q.music_noise}")

    out_root = resolve(cfg.paths.output)
    state_dir = resolve(cfg.paths.state_dir)
    device = cfg.runtime.device

    chunks = sorted(out_root.glob("*/chunks/*"))
    chunks = [c for c in chunks if c.suffix in (".wav", ".mp3")]
    if not chunks:
        warn(f"no chunks under {out_root} — run diarization first")
        return
    info(f"{len(chunks)} chunk(s) to score")

    scorers = []
    if q.utmos:       scorers.append(UTMOSScorer(device))
    if q.dnsmos:      scorers.append(DNSMOSScorer(device))
    if q.music_noise: scorers.append(MusicNoiseScorer(device))

    # resumable: cache per-chunk results so re-runs don't recompute
    cache_path = state_dir / "quality_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    records = []
    with progress() as bar:
        task = bar.add_task("[accent]scoring[/accent]", total=len(chunks))
        for c in chunks:
            key = str(c.relative_to(out_root))
            if key in cache:
                records.append(cache[key])
                bar.advance(task)
                continue
            wav, sr = sf.read(c, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            scores = {}
            for s in scorers:
                try:
                    scores.update(s.score(wav, sr))
                except Exception as e:
                    note(f"{s.name} failed on {key}: {e}")
            rec = {
                "chunk_path": key,
                "duration": round(len(wav) / sr, 3),
                "scores": scores,
                "overall": _overall(scores, q.weights),
            }
            cache[key] = rec
            records.append(rec)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False))
            bar.advance(task)

    # enrich with start/end from each audio's manifest
    _attach_timestamps(records, out_root)

    # write chunks.json (all) and chunks_clean.json (>= threshold)
    chunks_json = out_root / "chunks.json"
    chunks_json.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    # Filter on a single, decisive metric (default: dnsmos_ovrl). The blended
    # "overall" can be misled by side metrics (stoi/cleanliness) and let through
    # noisy chunks with laughter / multiple voices despite a low DNSMOS — so we
    # gate directly on the metric that actually reflects perceptual quality.
    thr = q.filter_threshold
    metric = getattr(q, "filter_metric", "dnsmos_ovrl")

    def _val(r):
        return r["scores"].get(metric) if metric != "overall" else r.get("overall")

    clean = [r for r in records if (_val(r) or 0) >= thr]
    (out_root / "chunks_clean.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2))

    # report hours
    total_h = sum(r["duration"] for r in records) / 3600
    clean_h = sum(r["duration"] for r in clean) / 3600
    vals = [_val(r) for r in records if _val(r) is not None]
    mean_metric = np.mean(vals) if vals else 0.0

    ok(f"wrote chunks.json ({len(records)}) and chunks_clean.json ({len(clean)})")
    info(f"filter: [accent]{metric} >= {thr}[/accent]  (mean {metric}: {mean_metric:.3f})")
    info(f"total audio:  [accent]{total_h:.2f} h[/accent]  ({len(records)} chunks)")
    info(f"clean:        [accent]{clean_h:.2f} h[/accent]  ({len(clean)} chunks)")


def _attach_timestamps(records, out_root: Path) -> None:
    """Pull start/end/speaker from each audio's manifest.csv into the records."""
    import csv
    index = {}
    for man in out_root.glob("*/manifest.csv"):
        sub = man.parent.name
        with open(man, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                index[f"{sub}/{row['chunk_path']}"] = row
    for r in records:
        m = index.get(r["chunk_path"])
        if m:
            r["speaker"] = m.get("speaker")
            r["start"] = float(m["start"])
            r["end"] = float(m["end"])
