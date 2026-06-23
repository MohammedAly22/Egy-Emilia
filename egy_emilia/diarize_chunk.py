"""
Single-speaker chunking pipeline for TTS data collection (Egyptian Arabic podcasts).

Goal: from one long audio file, produce a folder of WAV chunks where each chunk is
  - exactly ONE speaker (no overlap, no speaker change inside the chunk),
  - 3 to 30 seconds long,
  - cut at natural silences (no mid-word / sudden cuts),
  - clean enough to train a TTS on.

Strategy (why each stage exists):
  1. Silero VAD  -> finds *speech* regions, so we never cut mid-word and never
                    include long silence. Gives natural pause points for splitting.
  2. NeMo Sortformer (end-to-end diarizer, 2025) -> per-frame "who is speaking",
                    INCLUDING overlap. Much stronger than the old clustering
                    pipeline and than pyannote on non-English audio.
  3. Intersect    -> keep only stretches that are (a) inside VAD speech and
                    (b) a single speaker with ZERO overlap frames. Speaker changes
                    and overlap become hard boundaries (with a safety margin).
  4. Embedding purity check (TitaNet) -> optional final gate: a chunk whose internal
                    speaker embeddings are not tight is rejected (a 2nd speaker leaked
                    in). This is what gets purity close to perfect.

Configure everything in the CONFIG block at the bottom (__main__) — no CLI args.
Run on a GPU machine (H100). See README for exact commands.
"""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

SR = 16000              # Sortformer + Silero both want 16 kHz mono
FRAME = 0.08            # Sortformer frame = 80 ms


# --------------------------------------------------------------------------- #
# Config — all tunables live here; set them in __main__
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # paths
    audio: str                                  # input long audio file
    out_dir: str                                # output folder (chunks + manifests)

    # models
    diar_model: str = "nvidia/diar_streaming_sortformer_4spk-v2"

    # chunk length
    min_chunk_s: float = 3.0
    max_chunk_s: float = 30.0
    target_chunk_s: float = 20.0                # greedy splitter aims for this length

    # silero VAD — larger silence/pad => longer regions, no clipped word ends
    vad_threshold: float = 0.5
    min_silence_ms: int = 700                   # only a real pause ends a region
    speech_pad_ms: int = 250                    # keep word onsets/tails

    # diarization frame labelling
    diar_threshold: float = 0.5                 # speaker "active" if prob >= this
    boundary_margin_s: float = 0.20             # trimmed off each end of every run
    # bridge brief same-speaker interruptions (breath / "mhm" / momentary dropout)
    # so a single turn stays ONE long region instead of many short ones.
    bridge_gap_s: float = 0.6                   # silence gaps up to this are absorbed
    bridge_overlap_s: float = 0.0               # overlap frames up to this are absorbed (0 = never)
    # MERGE adjacent same-speaker regions separated only by silence (no other
    # speaker, no overlap) up to this gap. This is what makes a single-speaker
    # video produce long chunks: sentence pauses no longer fragment the turn.
    # The splitter still cuts the merged region at silences into target lengths.
    merge_gap_s: float = 8.0

    # tail safety: extra audio kept after each cut so words aren't clipped
    tail_pad_s: float = 0.15

    # output sampling rate / format (analysis runs at 16k; chunks written at this)
    write_sr: int = 24000
    write_format: str = "mp3"                   # "mp3" or "wav"
    mp3_bitrate: str = "192k"

    # TitaNet purity gate
    embedding_check: bool = True                # recommended for TTS
    embedding_threshold: float = 0.55           # reject if min intra-chunk cos < this

    @classmethod
    def from_dict(cls, audio: str, out_dir: str, d: dict) -> "Config":
        """Build from the central config's `diarization` block (extra keys ignored)."""
        fields = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        kw = {k: v for k, v in d.items() if k in fields}
        return cls(audio=audio, out_dir=out_dir, **kw)


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_audio_16k_mono(path: str) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


def write_chunk(seg: np.ndarray, sr: int, out_path: Path, fmt: str, mp3_bitrate: str):
    """Write a chunk as wav (soundfile) or mp3 (soundfile->wav then ffmpeg encode)."""
    if fmt == "wav":
        sf.write(out_path, seg, sr)
        return
    # mp3: encode via ffmpeg. Put the temp wav NEXT TO the output (same filesystem),
    # not in /tmp, to avoid cross-device / permission issues on mounted volumes.
    import subprocess
    tmp_wav = out_path.with_suffix(".tmp.wav")
    try:
        sf.write(tmp_wav, seg, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_wav),
             "-ar", str(sr), "-ac", "1", "-b:a", mp3_bitrate, str(out_path)],
            check=True,
        )
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()


# --------------------------------------------------------------------------- #
# Stage 1 — Silero VAD: speech regions [(start_s, end_s), ...]
# --------------------------------------------------------------------------- #
def run_vad(wav: np.ndarray, cfg: Config) -> List[Tuple[float, float]]:
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(wav),
        model,
        sampling_rate=SR,
        threshold=cfg.vad_threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=cfg.min_silence_ms,  # pause that ends a region
        speech_pad_ms=cfg.speech_pad_ms,             # small pad so edges aren't clipped
        return_seconds=True,
    )
    return [(t["start"], t["end"]) for t in ts]


# --------------------------------------------------------------------------- #
# Stage 2 — Sortformer diarization: per-frame speaker activity (T x S)
# --------------------------------------------------------------------------- #
def run_diarization(audio_path: str, cfg: Config):
    """Return (probs TxS float32, frame_sec). probs[t, s] = P(speaker s active at t)."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(cfg.diar_model)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Offline / high-accuracy streaming config (long chunks, full context).
    m = model.sortformer_modules
    m.chunk_len = 340
    m.chunk_right_context = 40
    m.fifo_len = 40
    m.spkcache_update_period = 300

    _, probs = model.diarize(
        audio=audio_path, batch_size=1, include_tensor_outputs=True
    )
    p = probs[0]
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    p = p.astype(np.float32)

    # Normalize to T x S (time, speakers). Here diarize() returns (1, T, S) e.g.
    # (1, 59983, 4): drop any leading singleton (batch) dims, then orient so the
    # large axis is time and the small axis is speakers.
    p = np.squeeze(p)
    if p.ndim != 2:
        raise ValueError(f"unexpected diarization output shape {p.shape}")
    if p.shape[0] < p.shape[1]:
        p = p.T
    return p, FRAME


# --------------------------------------------------------------------------- #
# Stage 3 — Intersect VAD + diarization into pure single-speaker regions
# --------------------------------------------------------------------------- #
def frame_labels(probs: np.ndarray, cfg: Config):
    """Per-frame label: speaker id if exactly one active, -1 if silence, -2 if overlap."""
    active = probs >= cfg.diar_threshold
    n_active = active.sum(axis=1)
    labels = np.full(probs.shape[0], -1, dtype=np.int64)  # default silence
    overlap = n_active >= 2
    single = n_active == 1
    labels[single] = active[single].argmax(axis=1)
    labels[overlap] = -2
    return labels


def bridge_gaps(labels: np.ndarray, frame_sec: float, cfg: Config):
    """Absorb short interruptions so one speaker turn stays a single long region.

    A gap of silence (-1) — or overlap (-2), if allowed — that sits *between two
    runs of the SAME speaker* and is short enough is relabelled to that speaker.
    This is the main lever for longer chunks: it stops breaths, back-channels and
    momentary diarizer dropouts from fragmenting a turn.
    """
    sil_max = int(round(cfg.bridge_gap_s / frame_sec))
    ov_max = int(round(cfg.bridge_overlap_s / frame_sec))
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] >= 0:  # inside a speaker run
            i += 1
            continue
        # gap run [i, j)
        j = i
        is_overlap = False
        while j < n and out[j] < 0:
            if out[j] == -2:
                is_overlap = True
            j += 1
        left = out[i - 1] if i > 0 else -1
        right = out[j] if j < n else -1
        glen = j - i
        limit = ov_max if is_overlap else sil_max
        if left >= 0 and left == right and glen <= limit:
            out[i:j] = left
        i = j
    return out


def pure_regions(labels: np.ndarray, frame_sec: float, cfg: Config):
    """Single-speaker regions, merged across silence-only gaps, margins trimmed.

    Steps:
      1. Find maximal runs of one speaker (frame index based).
      2. MERGE consecutive same-speaker runs when the gap between them is
         silence-ONLY (no different speaker / no overlap frame) and <= merge_gap_s.
         -> a single speaker's sentences stop fragmenting into ~3s pieces.
      3. Trim a safety margin off each merged region's ends and drop sub-min ones.

    Returns list of (start_s, end_s, speaker_id) in seconds.
    """
    # 1) maximal same-speaker runs, as [start_frame, end_frame, spk]
    runs = []
    n = len(labels)
    i = 0
    while i < n:
        spk = labels[i]
        if spk < 0:
            i += 1
            continue
        j = i
        while j < n and labels[j] == spk:
            j += 1
        runs.append([i, j, int(spk)])
        i = j

    # 2) merge across silence-only gaps
    merge_max = int(round(cfg.merge_gap_s / frame_sec))
    merged = []
    for run in runs:
        if merged:
            ps, pe, pspk = merged[-1]
            gap = labels[pe:run[0]]            # frames strictly between the two runs
            silence_only = bool(np.all(gap < 0)) and not bool(np.any(gap == -2))
            if run[2] == pspk and silence_only and (run[0] - pe) <= merge_max:
                merged[-1][1] = run[1]         # extend previous region over the gap
                continue
        merged.append(list(run))

    # 3) trim margins, convert to seconds, drop too-short
    margin = cfg.boundary_margin_s
    out = []
    for s_f, e_f, spk in merged:
        s = s_f * frame_sec + margin
        e = e_f * frame_sec - margin
        if e - s >= cfg.min_chunk_s:
            out.append((s, e, spk))
    return out


def vad_silence_points(vad: list, region_start: float, region_end: float):
    """Silence gaps (between VAD speech segments) inside [region_start, region_end].

    Used as natural split points so long regions are cut at pauses, not mid-word.
    Returns sorted list of gap midpoints (seconds).
    """
    pts = []
    prev_end = None
    for s, e in vad:
        if e <= region_start or s >= region_end:
            continue
        if prev_end is not None and s > prev_end:
            mid = 0.5 * (prev_end + s)
            if region_start < mid < region_end:
                pts.append(mid)
        prev_end = max(prev_end or 0, e)
    return sorted(pts)


def split_to_duration(region, vad, cfg: Config):
    """Split a (start, end, spk) region into target-length pieces at VAD silences.

    Aims for target_chunk_s (cutting at the pause closest to target, within
    [min, max]); only forces a cut at max if no pause exists. A region shorter
    than max stays whole. Cuts land on silences => never mid-word.
    """
    start, end, spk = region
    pieces = []
    cur = start
    pts = vad_silence_points(vad, start, end)
    while end - cur > cfg.max_chunk_s:
        window_lo = cur + cfg.min_chunk_s
        window_hi = cur + cfg.max_chunk_s
        target = cur + cfg.target_chunk_s
        cands = [p for p in pts if window_lo <= p <= window_hi]
        if cands:
            # pause nearest the target length (keeps chunks long but uniform)
            cut = min(cands, key=lambda p: abs(p - target))
        else:
            # no pause to cut on -> hard cut at max (rare; long monologue w/o pause)
            cut = window_hi
        pieces.append((cur, cut, spk))
        cur = cut
    if end - cur >= cfg.min_chunk_s:
        pieces.append((cur, end, spk))
    return pieces


# --------------------------------------------------------------------------- #
# Stage 4 — Optional embedding purity gate (TitaNet)
# --------------------------------------------------------------------------- #
class EmbeddingChecker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.model = None
        if enabled:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
            self.model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

    def is_pure(self, wav_seg: np.ndarray, threshold: float) -> bool:
        """Outlier-based single-speaker check.

        The diarizer already guarantees single-speaker (we keep only frames where
        exactly one speaker is active, drop overlap, trim margins). This gate is a
        light safety net for the rare case a different voice leaks in.

        Method: embed 3s windows, compute the chunk CENTROID, and reject only if a
        window is a clear OUTLIER from the centroid (cosine < threshold). Comparing
        to the centroid — not worst-pair min — tolerates the normal phonetic
        variation within one speaker that wrecked the old min-pairwise check.
        """
        if not self.enabled:
            return True
        win = int(3.0 * SR)          # longer window => stable embedding
        hop = int(1.5 * SR)
        if len(wav_seg) < win:
            return True              # too short to sub-check; duration gate passed
        embs = []
        for st in range(0, len(wav_seg) - win + 1, hop):
            seg = wav_seg[st:st + win]
            with torch.no_grad():
                t = torch.from_numpy(seg).unsqueeze(0)
                length = torch.tensor([seg.shape[0]])
                if torch.cuda.is_available():
                    t, length = t.cuda(), length.cuda()
                _, emb = self.model.forward(input_signal=t, input_signal_length=length)
                embs.append(torch.nn.functional.normalize(emb, dim=-1).squeeze(0))
        if len(embs) < 2:
            return True
        E = torch.stack(embs)
        centroid = torch.nn.functional.normalize(E.mean(dim=0, keepdim=True), dim=-1)
        sims = (E @ centroid.T).squeeze(1)        # each window vs centroid
        # reject only if the *most deviant* window is a clear outlier
        return float(sims.min()) >= threshold


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(cfg: Config):
    out = Path(cfg.out_dir)
    chunk_dir = out / "chunks"
    # NOTE: don't create chunk_dir yet. If a stage below crashes we must NOT leave
    # an empty chunks/ folder behind — the resume logic treats "folder exists but
    # empty" specially, and a stray empty dir would otherwise mask a failed audio.

    print("[1/4] loading audio @16k mono (for analysis) ...")
    wav = load_audio_16k_mono(cfg.audio)
    total_s = len(wav) / SR
    print(f"      duration: {total_s/60:.1f} min")

    # Original-SR signal that the final chunks are actually sliced from.
    import librosa
    wav_out, out_sr = librosa.load(cfg.audio, sr=cfg.write_sr, mono=True)
    wav_out = wav_out.astype(np.float32)
    print(f"      writing chunks at {out_sr} Hz")

    print("[2/4] Silero VAD ...")
    vad = run_vad(wav, cfg)
    print(f"      {len(vad)} speech regions")

    print("[3/4] Sortformer diarization ...")
    probs, frame_sec = run_diarization(cfg.audio, cfg)
    labels = frame_labels(probs, cfg)
    labels = bridge_gaps(labels, frame_sec, cfg)  # merge brief same-speaker gaps
    n_spk = int(labels[labels >= 0].max()) + 1 if (labels >= 0).any() else 0
    ov = float((labels == -2).mean()) * 100
    print(f"      detected up to {n_spk} speakers | overlap frames: {ov:.1f}%")

    regions = pure_regions(labels, frame_sec, cfg)
    if regions:
        rl = sorted(e - s for s, e, _ in regions)
        print(f"      {len(regions)} single-speaker regions | "
              f"region len s: min {rl[0]:.1f} median {rl[len(rl)//2]:.1f} max {rl[-1]:.1f}")
    else:
        print("      0 single-speaker regions")

    # Expand regions -> 3-30s pieces first, so the progress bar reflects real work.
    pieces = []
    for region in tqdm(regions, desc="splitting to 3-30s", unit="region"):
        pieces.extend(split_to_duration(region, vad, cfg))
    if pieces:
        pl = sorted(e - s for s, e, _ in pieces)
        print(f"      {len(pieces)} pieces after split | "
              f"piece len s: min {pl[0]:.1f} median {pl[len(pl)//2]:.1f} max {pl[-1]:.1f}")

    print(f"[4/4] purity gate + writing {len(pieces)} candidate chunks ...")
    chunk_dir.mkdir(parents=True, exist_ok=True)  # create only now that we have work
    checker = EmbeddingChecker(cfg.embedding_check)

    rows = []
    idx = 0
    rejected = 0
    errored = 0
    for (s, e, spk) in tqdm(pieces, desc="writing chunks", unit="chunk"):
        try:
            # purity check on the 16k analysis signal
            seg16 = wav[int(s * SR):int(e * SR)]
            if checker.enabled and not checker.is_pure(seg16, cfg.embedding_threshold):
                rejected += 1
                continue
            # write from the original-SR signal, with a tail pad so word ends aren't clipped
            e_pad = min(e + cfg.tail_pad_s, total_s)
            seg_out = wav_out[int(s * out_sr):int(e_pad * out_sr)]
            name = f"chunk_{idx + 1:05d}_spk{spk}.{cfg.write_format}"
            write_chunk(seg_out, out_sr, chunk_dir / name, cfg.write_format, cfg.mp3_bitrate)
        except Exception as ex:
            # one bad chunk must never abort the whole audio
            errored += 1
            if errored <= 3:
                print(f"      ! chunk write failed at {s:.1f}-{e:.1f}s: {ex}")
            continue
        idx += 1
        rows.append({
            "chunk_path": f"chunks/{name}",
            "speaker": f"speaker_{spk}",
            "start": round(s, 3),
            "end": round(e_pad, 3),
            "duration": round(e_pad - s, 3),
        })
    if errored:
        print(f"      {errored} chunk(s) failed to write (continued anyway)")

    # write CSV + JSONL manifest
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chunk_path", "speaker", "start", "end", "duration"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dur = sum(r["duration"] for r in rows)
    print(f"\nDONE: {len(rows)} clean chunks, {dur/60:.1f} min kept "
          f"({rejected} rejected by purity gate).")
    print(f"  audio:    {chunk_dir}")
    print(f"  manifest: {out/'manifest.csv'}")
    for spk, c in sorted(Counter(r['speaker'] for r in rows).items()):
        sd = sum(r['duration'] for r in rows if r['speaker'] == spk)
        print(f"    {spk}: {c} chunks, {sd/60:.1f} min")


# --------------------------------------------------------------------------- #
# CONFIG — edit paths / tunables here, then run:  python diarize_chunk.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = Config(
        audio="long_audios/audio_4axSKMfXHlE.wav",
        out_dir="diar_out",

        # length window (chunks aim for target, allowed up to max)
        min_chunk_s=3.0,
        target_chunk_s=20.0,
        max_chunk_s=30.0,

        # LONGER chunks: bridge brief same-speaker gaps so a turn stays one region.
        # Raise bridge_gap_s for even longer chunks (e.g. 1.0); set bridge_overlap_s
        # > 0 only if you accept tiny back-channel overlaps.
        bridge_gap_s=0.6,
        bridge_overlap_s=0.0,

        # no clipped word ends
        speech_pad_ms=250,
        tail_pad_s=0.15,
        min_silence_ms=700,

        # output at the original sampling rate
        write_sr=24000,

        # purity (raise embedding_threshold / boundary_margin_s if you hear 2 voices)
        embedding_check=True,
        embedding_threshold=0.55,
        boundary_margin_s=0.20,

        # VAD / diarization sensitivity
        vad_threshold=0.5,
        diar_threshold=0.5,
    )
    run(cfg)
