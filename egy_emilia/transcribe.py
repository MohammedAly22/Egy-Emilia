"""Stage 5 — transcribe the CLEAN chunks (Egyptian Arabic + code-switching).

ASR backend is pluggable (config.asr.backend):
  • "qwencleo"  — mohammedaly22/QwenCleo-ASR (Qwen3-ASR-1.7B), purpose-built for
                  Egyptian Arabic + Arabic/English code-switching. DEFAULT.
  • "whisper"   — Whisper large-v3 family; keeps code-switched English in Latin script.
  • "egyptalk"  — NAMAA-Space/EgypTalk-ASR-v2, NeMo FastConformer (Arabic script only).
  • "seamless"  — facebook/seamless-m4t-v2-large, Egyptian Arabic (arz).

Output: transcriptions are written back into chunks_clean.json (field "text"),
and a transcripts.json is also written. Resumable per chunk.
"""

import json

from .config import resolve
from .ui import banner, info, ok, progress, warn


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class QwenCleoBackend:
    """QwenCleo-ASR (Qwen3-ASR-1.7B) — Egyptian Arabic + code-switching. Default."""

    def __init__(self, cfg):
        from qwencleo_asr import QwenCleoASR
        # Constructor loads mohammedaly22/QwenCleo-ASR by default.
        self.asr = QwenCleoASR()
        # "Arabic" anchors the dialect; English terms are kept in Latin script.
        self.language = None if cfg.asr.qwencleo_language in (None, "auto") \
            else cfg.asr.qwencleo_language
        self.batch_size = cfg.asr.batch_size

    def transcribe(self, paths: list[str]) -> list[str]:
        # batch -> list of result objects, each with .text
        results = self.asr.transcribe(list(paths), language=self.language)
        if not isinstance(results, (list, tuple)):
            results = [results]
        return [(getattr(r, "text", "") or "").strip() for r in results]



class EgypTalkBackend:
    """NeMo FastConformer for Egyptian Arabic."""

    def __init__(self, cfg):
        from nemo.collections.asr.models import ASRModel
        import torch

        model_id = cfg.asr.egyptalk_model
        # This repo ships a .nemo whose name != repo name, so from_pretrained's
        # restore step can't find model_config.yaml. Download the actual .nemo
        # file and restore_from it directly.
        try:
            self.model = ASRModel.from_pretrained(model_id)
        except FileNotFoundError:
            from huggingface_hub import hf_hub_download, list_repo_files
            nemo_files = [f for f in list_repo_files(model_id) if f.endswith(".nemo")]
            if not nemo_files:
                raise
            local = hf_hub_download(repo_id=model_id, filename=nemo_files[0])
            self.model = ASRModel.restore_from(local)

        self.model.eval()
        if cfg.runtime.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.batch_size = cfg.asr.batch_size

    def transcribe(self, paths: list[str]) -> list[str]:
        out = self.model.transcribe(paths, batch_size=self.batch_size)
        # NeMo may return list[str] or list[Hypothesis]
        return [getattr(o, "text", o) for o in out]


class SeamlessBackend:
    """Meta Seamless M4T v2 — Egyptian Arabic (arz)."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        self.processor = AutoProcessor.from_pretrained(cfg.asr.seamless_model)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(cfg.asr.seamless_model)
        self.device = "cuda" if (cfg.runtime.device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model.to(self.device)

    def transcribe(self, paths: list[str]) -> list[str]:
        import soundfile as sf
        import torch
        texts = []
        for p in paths:
            wav, sr = sf.read(p, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            inputs = self.processor(audios=wav, sampling_rate=sr, return_tensors="pt").to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**inputs, tgt_lang="arz")[0]
            texts.append(self.processor.decode(ids, skip_special_tokens=True))
        return texts


class WhisperCSBackend:
    """Whisper (large-v3 family) for Arabic + English code-switching.

    Key trick for code-switching: run task='transcribe' WITHOUT forcing a language,
    so the model emits English words in Latin script instead of transliterating or
    translating them. Default model is a code-switching-tuned Whisper.
    """

    def __init__(self, cfg):
        import torch
        from transformers import pipeline
        self.device = 0 if (cfg.runtime.device == "cuda" and torch.cuda.is_available()) else -1
        self.dtype = torch.float16 if self.device == 0 else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=cfg.asr.whisper_model,
            torch_dtype=self.dtype,
            device=self.device,
            chunk_length_s=30,
        )
        # language=arabic anchors the decoder (avoids empty output); task=transcribe
        # (not translate) keeps code-switched English words verbatim in Latin script.
        self.gen_kwargs = {"task": "transcribe", "language": "arabic"}
        self.batch_size = cfg.asr.batch_size

    def transcribe(self, paths: list[str]) -> list[str]:
        # Decode audio OURSELVES (librosa) and pass raw 16k arrays. This bypasses
        # transformers' torchcodec audio loader, which tries to dlopen a CUDA-13
        # lib (libnvrtc.so.13) and spams errors on a CUDA-12 box.
        import librosa
        import numpy as np
        inputs = []
        for p in paths:
            wav, _ = librosa.load(p, sr=16000, mono=True)
            inputs.append({"raw": wav.astype(np.float32), "sampling_rate": 16000})
        outs = self.pipe(inputs, batch_size=self.batch_size,
                         generate_kwargs=self.gen_kwargs)
        if isinstance(outs, dict):
            outs = [outs]
        return [(o.get("text") or "").strip() for o in outs]


def _make_backend(cfg):
    b = cfg.asr.backend.lower()
    if b == "qwencleo":
        return QwenCleoBackend(cfg)
    if b == "whisper":
        return WhisperCSBackend(cfg)
    if b == "egyptalk":
        return EgypTalkBackend(cfg)
    if b == "seamless":
        return SeamlessBackend(cfg)
    raise ValueError(f"unknown asr.backend: {cfg.asr.backend!r}")


# --------------------------------------------------------------------------- #
def run(cfg) -> None:
    banner("Stage 5 · Transcription (Egyptian Arabic + code-switching)",
           f"backend = {cfg.asr.backend}")

    out_root = resolve(cfg.paths.output)
    clean_path = out_root / "chunks_clean.json"
    if not clean_path.exists():
        warn("chunks_clean.json not found — run quality scoring first")
        return

    records = json.loads(clean_path.read_text())
    todo = [r for r in records if not r.get("text")]
    info(f"{len(records)} clean chunk(s); {len(todo)} need transcription")
    if not todo:
        ok("all clean chunks already transcribed")
        return

    with console_status_loading():
        backend = _make_backend(cfg)

    bs = cfg.asr.batch_size
    by_path = {r["chunk_path"]: r for r in records}
    with progress() as bar:
        task = bar.add_task("[accent]transcribing[/accent]", total=len(todo))
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            paths = [str(out_root / r["chunk_path"]) for r in batch]
            try:
                texts = backend.transcribe(paths)
            except Exception as e:
                warn(f"batch failed: {e}")
                texts = [""] * len(batch)
            for r, t in zip(batch, texts):
                by_path[r["chunk_path"]]["text"] = (t or "").strip()
            # checkpoint after each batch so an interrupt loses at most one batch
            clean_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            bar.advance(task, advance=len(batch))

    (out_root / "transcripts.json").write_text(
        json.dumps([{"chunk_path": r["chunk_path"], "text": r.get("text", "")}
                    for r in records], ensure_ascii=False, indent=2))
    ok(f"transcribed {len(todo)} chunk(s) → chunks_clean.json + transcripts.json")


def console_status_loading():
    from .ui import console
    return console.status("[info]loading ASR model …[/info]", spinner="dots")
