#!/usr/bin/env python3
"""EGY-Emilia · validation / debug harness for a SINGLE audio.

Runs the diarization+chunking engine on one audio and prints a full breakdown of
every stage so we can see exactly where chunks are gained/lost:

  • speakers detected, overlap %
  • VAD speech regions
  • single-speaker regions (after merge) + length stats
  • pieces after 3-30s split + length stats
  • purity-gate accept/reject counts + the similarity distribution that drives them
  • final kept chunks: count, total minutes, per-speaker, duration histogram

Edit AUDIO below, then:  python validate_audio.py
(No CLI args — uses config.yaml for the diarization settings.)
"""

import statistics as st
from collections import Counter

import numpy as np

from egy_emilia import diarize_chunk as dc
from egy_emilia.config import load_config, resolve
from egy_emilia.ui import banner, console, info, ok, warn

# ---- pick the audio to debug -------------------------------------------------
AUDIO = "input_audios/ekDVHRwB2hU.mp3"
# -----------------------------------------------------------------------------


def _stats(name, lengths):
    if not lengths:
        warn(f"{name}: 0 items")
        return
    lengths = sorted(lengths)
    info(f"{name}: n={len(lengths)} "
         f"min={lengths[0]:.1f} median={st.median(lengths):.1f} "
         f"mean={st.mean(lengths):.1f} max={lengths[-1]:.1f}")


def _hist(durs):
    b = Counter()
    for d in durs:
        k = "3-5" if d < 5 else "5-10" if d < 10 else "10-20" if d < 20 else "20-30" if d <= 30 else ">30"
        b[k] += 1
    info(f"duration buckets: {dict(b)}")


def main():
    cfg = load_config()
    diar = dc.Config.from_dict(str(resolve(AUDIO)), "output/_validate", vars(cfg.diarization))

    banner("EGY-Emilia · Validation", AUDIO)

    # --- stage 1: load ---
    wav = dc.load_audio_16k_mono(diar.audio)
    total_s = len(wav) / dc.SR
    info(f"duration: {total_s/60:.1f} min @ analysis 16k")

    # --- stage 2: VAD ---
    vad = dc.run_vad(wav, diar)
    speech_s = sum(e - s for s, e in vad)
    info(f"VAD: {len(vad)} speech regions, {speech_s/60:.1f} min speech "
         f"({100*speech_s/total_s:.0f}% of audio)")

    # --- stage 3: diarization ---
    probs, frame_sec = dc.run_diarization(diar.audio, diar)
    labels = dc.frame_labels(probs, diar)
    labels = dc.bridge_gaps(labels, frame_sec, diar)
    n_spk = int(labels[labels >= 0].max()) + 1 if (labels >= 0).any() else 0
    ov = float((labels == -2).mean()) * 100
    sil = float((labels == -1).mean()) * 100
    console.rule("[accent]diarization[/accent]")
    info(f"speakers detected: {n_spk}")
    info(f"frames: speech {100-ov-sil:.1f}% | overlap {ov:.1f}% | silence {sil:.1f}%")
    per_spk = Counter(int(x) for x in labels if x >= 0)
    for s, c in sorted(per_spk.items()):
        info(f"  speaker_{s}: {c*frame_sec/60:.1f} min of single-speaker frames")

    # --- regions ---
    regions = dc.pure_regions(labels, frame_sec, diar)
    console.rule("[accent]single-speaker regions (after merge)[/accent]")
    _stats("regions", [e - s for s, e, _ in regions])

    # --- split ---
    pieces = []
    for r in regions:
        pieces.extend(dc.split_to_duration(r, vad, diar))
    console.rule("[accent]pieces after 3-30s split[/accent]")
    _stats("pieces", [e - s for s, e, _ in pieces])
    _hist([e - s for s, e, _ in pieces])

    # --- purity gate: show the similarity distribution that drives accept/reject ---
    console.rule("[accent]purity gate (TitaNet)[/accent]")
    if not diar.embedding_check:
        warn("embedding_check disabled — all pieces would be kept")
    checker = dc.EmbeddingChecker(diar.embedding_check)
    wav_out, out_sr = wav, dc.SR  # purity runs on 16k anyway
    sims = []
    accept = reject = 0
    for (s, e, spk) in pieces:
        seg = wav[int(s * dc.SR):int(e * dc.SR)]
        sim = _centroid_min_sim(checker, seg)
        if sim is not None:
            sims.append(sim)
        keep = checker.is_pure(seg, diar.embedding_threshold) if checker.enabled else True
        if keep:
            accept += 1
        else:
            reject += 1
    info(f"threshold = {diar.embedding_threshold}")
    if sims:
        ss = sorted(sims)
        info(f"window-vs-centroid min-sim distribution: "
             f"p05={np.percentile(ss,5):.2f} p25={np.percentile(ss,25):.2f} "
             f"median={np.median(ss):.2f} p75={np.percentile(ss,75):.2f} max={ss[-1]:.2f}")
        info("  (single speaker usually ~0.85+; a real 2nd voice dips ~0.5-0.7)")
    accept_pct = 100 * accept / max(1, len(pieces))
    ok(f"ACCEPT {accept} ({accept_pct:.0f}%)  •  REJECT {reject}")
    if accept == 0:
        warn("0 accepted — threshold is too high for this audio; lower embedding_threshold")

    # --- final summary ---
    console.rule("[accent]final[/accent]")
    kept = [(s, e, spk) for (s, e, spk) in pieces
            if (not checker.enabled) or checker.is_pure(wav[int(s*dc.SR):int(e*dc.SR)], diar.embedding_threshold)]
    durs = [e - s for s, e, _ in kept]
    ok(f"KEPT {len(kept)} chunks, {sum(durs)/60:.1f} min")
    if durs:
        _stats("kept", durs)
        _hist(durs)
        spk_min = Counter()
        for s, e, spk in kept:
            spk_min[spk] += e - s
        for spk, m in sorted(spk_min.items()):
            info(f"  speaker_{spk}: {m/60:.1f} min")


def _centroid_min_sim(checker, wav_seg):
    """Replicate the gate's internal min-sim so we can SEE the distribution."""
    if not checker.enabled:
        return None
    import torch
    win = int(3.0 * dc.SR)
    hop = int(1.5 * dc.SR)
    if len(wav_seg) < win:
        return None
    embs = []
    for stp in range(0, len(wav_seg) - win + 1, hop):
        seg = wav_seg[stp:stp + win]
        with torch.no_grad():
            t = torch.from_numpy(seg).unsqueeze(0)
            length = torch.tensor([seg.shape[0]])
            if torch.cuda.is_available():
                t, length = t.cuda(), length.cuda()
            _, emb = checker.model.forward(input_signal=t, input_signal_length=length)
            embs.append(torch.nn.functional.normalize(emb, dim=-1).squeeze(0))
    if len(embs) < 2:
        return None
    E = torch.stack(embs)
    centroid = torch.nn.functional.normalize(E.mean(dim=0, keepdim=True), dim=-1)
    return float((E @ centroid.T).squeeze(1).min())


if __name__ == "__main__":
    main()
