"""Cohere Transcribe (Arabic) ASR worker — runs in its OWN Python env.

cohere-transcribe needs transformers>=5.4, but NeMo / qwen-asr need 4.57.x, so
they cannot share one env. transcribe.CohereBackend starts this file with the
interpreter from asr.cohere_python (created by scripts/setup_cohere_env.sh) and
talks to it over stdin/stdout, one JSON object per line:

  -> {"paths": ["/abs/chunk1.mp3", ...]}
  <- {"texts": ["...", ...]}          or   {"error": "..."}

Nothing else may be written to stdout; all logging goes to stderr.
Standalone on purpose: it imports nothing from egy_emilia.
"""

import argparse
import json
import subprocess
import sys
import traceback

import numpy as np

SR = 16000                              # the model only accepts 16 kHz


def load_16k_mono(path: str) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 with ffmpeg (already required
    by the pipeline), so this env needs no audio libraries of its own."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def main() -> None:
    proto = sys.stdout                  # protocol channel
    sys.stdout = sys.stderr             # any library print() goes to the log

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--language", default="ar")
    ap.add_argument("--punctuation", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = CohereAsrForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(device).eval()

    def reply(obj):
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    reply({"ready": True, "device": device})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            paths = json.loads(line)["paths"]
            audio = [load_16k_mono(p) for p in paths]
            inputs = processor(audio=audio, sampling_rate=SR, return_tensors="pt",
                               language=args.language, punctuation=bool(args.punctuation))
            # clips > ~30 s are split by the feature extractor; this maps pieces back
            chunk_index = inputs.pop("audio_chunk_index", None)
            inputs = inputs.to(device, dtype=dtype)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            texts = [processor.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
            if chunk_index is not None and len(chunk_index) != len(paths):
                texts = processor._reassemble_chunk_texts(texts, chunk_index, " ")
            reply({"texts": texts})
        except Exception as e:
            traceback.print_exc()
            reply({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
