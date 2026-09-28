<div align="center">

# 🎙️ EGY-Emilia

### End-to-end pipeline for building **clean, single-speaker** Egyptian-Arabic TTS datasets

*From a list of YouTube links to quality-scored, transcribed, training-ready audio chunks — one command.*

![python](https://img.shields.io/badge/python-3.11-blue)
![torch](https://img.shields.io/badge/torch-2.7.1%20cu126-ee4c2c)
![nemo](https://img.shields.io/badge/NeMo-2.7.3-76b900)
![gpu](https://img.shields.io/badge/GPU-H100-success)

</div>

---

## ✨ What it does

EGY-Emilia turns raw long-form audio into a **TTS-grade corpus** where every chunk is
**one speaker, no overlap, 3–30 s, cleanly cut, loudness-normalized, quality-scored,
and transcribed** (Egyptian Arabic with English **code-switching** preserved).

```
🔗 sources.txt ─▶ ⬇️ download ─▶ ✂️ diarize+chunk ─▶ 🔊 loudness ─▶ 📊 quality ─▶ 📝 transcribe ─▶ 🤗 publish
                                                                            │
                                                          chunks.json ◀─────┤
                                                    chunks_clean.json ◀──────┘  (+ text)
```

| # | Stage | What happens | Output |
|---|-------|--------------|--------|
| 1️⃣ | **Download** | yt-dlp expands channels/playlists/videos → audio at target SR/format | `input_audios/*.mp3` |
| 2️⃣ | **Diarize + Chunk** | Silero VAD ∩ Sortformer ∩ TitaNet purity gate → pure single-speaker chunks | `output/<id>/chunks/` |
| 3️⃣ | **Loudness** | EBU R128 normalization (ffmpeg `loudnorm`) | chunks normalized in place |
| 4️⃣ | **Quality** | per-chunk UTMOS + DNSMOS + music/noise → `overall` score & filtering | `chunks.json`, `chunks_clean.json` |
| 5️⃣ | **Transcribe** | Egyptian-Arabic ASR w/ code-switching on **clean** chunks | `text` in `chunks_clean.json` |
| 6️⃣ | **Publish** | clean + transcribed chunks → **private** HuggingFace dataset | `hf_dataset/` → `huggingface.co/datasets/<repo_id>` |

> 🧩 **Everything is driven by [`config.yaml`](config.yaml).** No CLI args anywhere.
> 🔁 **Every stage is resumable** — interrupt any time, re-run, and it continues where it stopped.

---

## 🧠 Why this design (vs. plain pyannote)

Single-speaker **purity** is the hard constraint for TTS, so no single model is trusted alone:

| Stage | Tool | Job |
|-------|------|-----|
| VAD | **Silero VAD** | speech/silence edges → no mid-word cuts, natural split points |
| Diarization | **NeMo Sortformer** (`diar_streaming_sortformer_4spk-v2`) | per-frame *who*, **including overlap**. End-to-end → far better than pyannote on Arabic |
| Intersection | *(our logic)* | keep only runs that are in-speech **and** single-speaker with **zero overlap**; bridge brief same-speaker gaps for longer chunks |
| Purity gate | **TitaNet** | re-embed each chunk; reject if a 2nd voice leaked in |

---

## 🚀 Setup (H100 pod)

> 📁 Mount: `/mnt/storage/tts/m.aly` · Run from: `/home/workspace/m.aly/test asr diarization`

> 🟢 **Python 3.11** — NeMo pulls `transformers~=4.57` → `tokenizers>=0.22`, which has **no wheels for 3.13**.
> 3.11 has the widest wheel coverage and avoids `ResolutionImpossible`.

> 🎛️ **CUDA note** — the pod's `nvidia-smi` shows `CUDA 12.2`, but CUDA 12.x is **forward-compatible
> across minor versions** (driver only needs to beat `525.60.13`). PyTorch dropped `cu121` after 2.5.1,
> so torch 2.7 comes from the **`cu126`** index — and runs fine here.

```bash
cd "/home/workspace/m.aly/test asr diarization"

# 1) conda env (Python 3.11)
conda create -n egy python=3.11 -y
conda activate egy
pip install --upgrade pip setuptools wheel

# 2) PyTorch 2.7.1 + torchaudio. Pick the index for your GPU:
#    H100 / A100 / RTX 30xx-40xx      -> cu126
#    Blackwell: RTX 50xx / B200 / RTX PRO 6000 -> cu128  (cu126 fails with
#    "CUDA error: no kernel image is available for execution on the device")
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 3) pipeline deps
pip install -r requirements.txt

# 4) NeMo (Sortformer diarizer + TitaNet + EgypTalk ASR backend)
pip install "nemo_toolkit[asr]==2.7.3"

# 5) ffmpeg must be on PATH (download/encode + loudnorm)
conda install -c conda-forge ffmpeg -y   # or: apt-get install -y ffmpeg

# 6) deno — yt-dlp needs a JS runtime to solve YouTube's player challenges
conda install -c conda-forge deno -y

# 6b) OPTIONAL PO-token provider (only if diagnose_youtube.py shows you need it;
#     then set download.pot_provider: true)
# bash scripts/setup_pot_provider.sh

# 7) HuggingFace login (for the publish stage; token needs WRITE access)
hf auth login                            # or: export HF_TOKEN=hf_xxx
```

---

## 🛡️ YouTube on RunPod (no cookies needed)

YouTube bot-checks datacenter IPs (**"Sign in to confirm you're not a bot"**). What gets through
on RunPod is **plain yt-dlp**: no JavaScript runtime, so yt-dlp only uses the `visionos` player
client, and no plugins. That is the default:

```yaml
download:
  js_runtime: false      # don't let yt-dlp use deno / web clients (those get bot-checked)
  pot_provider: false    # don't load the bgutil PO-token plugin
```

The download stage enforces this even when deno or the plugin is installed in the env. It also
paces requests (`sleep_min_s` / `sleep_max_s`, 10–30 s). YouTube still rate-limits an IP after a
burst of downloads, so after `bot_abort_after` consecutive blocks the stage **pauses**
(`cooldown_minutes`: 10 → 20 → 40 → 60 min, reset after every success), then retries the blocked
videos with a fresh session. It only gives up if the block outlasts every cooldown. Finished videos
are checkpointed. For long runs, prefer a terminal that survives disconnects:
`nohup python download.py > download.log 2>&1 &` (watch with `tail -f download.log`). Single-video links are parsed locally, with no request to YouTube.

### If the pod's IP is still blocked

PO tokens help a lot, but YouTube can still hard-block a particular datacenter IP. Fallbacks,
none of which use cookies:

0. **Find a player client that still works from this IP** (free, 1 minute; tests with your config's settings):
   ```bash
   python scripts/diagnose_youtube.py
   ```
   It tries each YouTube client (tv, web_embedded, android_vr, …). If one works, it prints an
   `extractor_args:` line to paste under `download:` in `config.yaml`.
1. **Download at home, process on the pod** (always works: home IPs aren't blocked).
   On your PC (Python + ffmpeg + deno installed):
   ```bash
   git clone https://github.com/MohammedAly22/Egy-Emilia && cd Egy-Emilia
   pip install -U "yt-dlp[default]" rich pyyaml
   python download.py                     # → input_audios/*.mp3
   runpodctl send input_audios            # prints a one-time code
   ```
   On the pod, in the repo root: `runpodctl receive <code>`, then `python run_pipeline.py`.
   Already-present audios are skipped, so the download stage just fetches anything missing.
   (`runpodctl` is preinstalled on pods; for Windows get it from
   https://github.com/runpod/runpodctl/releases.)
2. **New IP:** stop the pod and start one in another region, then re-run.
3. **Residential proxy:** set `download.proxy: "http://user:pass@host:port"` in `config.yaml`.

`download.cookies_file` still exists as an optional last resort, but it is `null` by default.

---

## ⚙️ Configure

Open [`config.yaml`](config.yaml) and set what you need. Highlights:

```yaml
download:
  audio_format: "mp3"      # 🎵 mp3 (smaller) or wav
  sample_rate: 24000       # 🎚️ default 24 kHz
  channels: 1              # mono

quality:
  filter_threshold: 2.8    # ✅ chunks with overall >= this → chunks_clean.json

asr:
  backend: "qwencleo"      # 🗣️ "qwencleo" | "cohere" | "whisper" | "egyptalk" | "seamless"
  preserve_english: true   # 🔤 keep code-switched English terms in Latin script

publish:
  repo_id: "mohammedaly22/Egy-Emilia"   # 🤗 <user-or-org>/<dataset-name>
  private: true                         # 🔒 only you can see it
```

Then add your links to [`sources.txt`](sources.txt) (channels, playlists, or videos — any mix).

---

## ▶️ Run

**Notebook, recommended on RunPod:** [`pipeline.ipynb`](pipeline.ipynb)

- **Part A** runs every stage on **one sample video** (in `sample/`, separate from the real data).
  It then checks the results (24 kHz mono, chunk lengths, scores, filter, transcripts) and plays
  kept and rejected chunks, so you can hear whether splitting, filtering and ASR are right.
- **Part B** is the full run, **one cell per stage** (download → diarize → loudness → quality →
  transcribe → summary → publish), each with its own progress. It won't start until Part A passes.

Use the `egy` env as the kernel (one-time):
```bash
conda activate egy
pip install ipykernel ipywidgets
python -m ipykernel install --user --name egy --display-name "Python (egy)"
```

**Whole pipeline from the terminal:**
```bash
python run_pipeline.py
```

**One stage at a time** — edit `STAGE` at the top of `run_stage.py`, then:
```bash
python run_stage.py        # STAGE = "download" | "diarize" | "loudness" | "quality" | "transcribe" | "publish"
```

**Just downloading:**
```bash
python download.py
```

You'll get colored, live **progress bars with spinners** (via `rich`) for every stage. 🌈

---

## 📦 Output layout

```
output/
├── <audio_id>/
│   ├── chunks/                 🎧 chunk_00001_spk0.mp3, ...
│   ├── manifest.csv            path, speaker, start, end, duration
│   └── manifest.jsonl
├── chunks.json                 📊 ALL chunks + duration + quality sub-scores + overall
├── chunks_clean.json           ✅ chunks with overall >= filter_threshold (+ transcriptions)
└── transcripts.json            📝 path → text
```

Example `chunks.json` record:
```json
{
  "chunk_path": "audio_4axSKMfXHlE/chunks/chunk_00007_spk1.mp3",
  "speaker": "speaker_1", "start": 41.2, "end": 61.6, "duration": 20.4,
  "scores": { "utmos": 4.12, "dnsmos_ovrl": 3.71, "cleanliness": 0.97 },
  "overall": 3.83,
  "text": "أنا شغال على الـ startup دي من سنتين"
}
```

At the end of **Stage 4** you'll see:
```
ℹ total audio:  12.40 h  (3812 chunks)
ℹ clean (>= 2.8): 9.10 h  (2790 chunks)
```

---

## 🤗 Publishing to HuggingFace (private)

The last pipeline stage uploads every chunk in `chunks_clean.json` **that has a
transcription** to `publish.repo_id` as a **private** dataset (visibility is enforced
even if the repo already exists). It is laid out as an HF *audiofolder*:

```
hf_dataset/                       (hardlinks to output/, no extra disk)
├── README.md                     dataset card
├── metadata.jsonl                file_name, text, duration, speaker, source_id, start, end, scores, overall
└── audio/<video_id>/chunk_*.mp3
```

The upload uses `upload_large_folder`, so it is resumable: if it is interrupted, re-run
`publish`. Load the dataset with:

```python
from datasets import load_dataset
ds = load_dataset("mohammedaly22/Egy-Emilia", split="train", token=True)
```

---

## 🔁 Resuming

Each stage keeps a checkpoint under `.state/`:

| Stage | Resume mechanism |
|-------|------------------|
| download | `download.json` (a video is marked done only once its audio file exists) |
| diarize | per-audio `diarize.json` (skips finished audios) |
| loudness | per-chunk `loudness.json` |
| quality | `quality_cache.json` (per-chunk scores) |
| transcribe | checkpoints `chunks_clean.json` after every batch |
| publish | `upload_large_folder` resumes from its cache in `hf_dataset/.cache` |

Interrupt with `Ctrl-C` and just re-run — finished work is skipped. 🟢

---

## 🗣️ ASR backends (pluggable)

Egyptian Arabic + code-switching is genuinely hard, and generic/fine-tuned Whisper
variants underperform on it. EGY-Emilia keeps ASR **pluggable** via `asr.backend`:

- **`qwencleo`** *(default)* — [`mohammedaly22/QwenCleo-ASR`](https://huggingface.co/mohammedaly22/QwenCleo-ASR),
  a **Qwen3-ASR-1.7B** model purpose-built for Egyptian Arabic + Arabic/English
  code-switching. Near-perfect on podcasts; keeps English in Latin script.
- **`whisper`** — Whisper large-v3 family; keeps code-switched English in Latin script.
- **`egyptalk`** — `NAMAA-Space/EgypTalk-ASR-v2`, NeMo FastConformer (Arabic script only).
- **`seamless`** — `facebook/seamless-m4t-v2-large`, Egyptian Arabic (`arz`).
- **`cohere`** — [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026),
  2B params, Arabic + dialects + Arabic/English code-switching, with punctuation. Runs in its own env (see below).

Switch via `asr.backend` in the config — no code changes.

### Installing the default (QwenCleo-ASR)

```bash
# torch is already installed (Setup step 2). Then:
pip install qwencleo-asr --no-deps
pip install "qwen-asr>=0.0.6"
pip install "transformers==4.57.6"
```
`--no-deps` keeps qwencleo-asr from re-resolving torch/transformers against our pinned
stack. First run downloads `mohammedaly22/QwenCleo-ASR` from HuggingFace.

> ⚠️ **transformers must be ≥ 4.57.3** (use `4.57.6`, which qwen-asr pins and NeMo accepts).
> NeMo may install an older 4.57.x, which makes every batch fail with
> `check_model_inputs() missing 1 required positional argument: 'func'`.

### Installing Cohere Transcribe Arabic

It needs **transformers ≥ 5.4**, which can't coexist with NeMo/QwenCleo's 4.57.x. So it gets
its own small venv that **reuses the egy env's torch** (no second torch download). The pipeline
talks to it through a worker process (`egy_emilia/asr_cohere_worker.py`):

```bash
conda activate egy
bash scripts/setup_cohere_env.sh      # creates .venvs/cohere + downloads the model (~4 GB)
```
Then set `asr.backend: "cohere"` in `config.yaml`. Options: `cohere_language` (`ar`/`en`),
`cohere_punctuation`. The worker's log is in `.state/cohere_worker.log`.

> ⚠️ Model availability shifts; if a checkpoint 404s, swap it in `config.yaml`
> (`asr.egyptalk_model` / `asr.seamless_model` / `asr.whisper_model`).

---

## 🧰 Tuning for longer / purer chunks

| Want | Change in `config.yaml` |
|------|--------------------------|
| 📏 Longer chunks | ↑ `diarization.bridge_gap_s` (e.g. `1.0`), ↑ `target_chunk_s` |
| 🧼 Purer chunks | ↑ `embedding_threshold` (e.g. `0.6`), ↑ `boundary_margin_s` |
| ✂️ No clipped words | ↑ `speech_pad_ms`, ↑ `tail_pad_s` |
| ✅ Stricter clean set | ↑ `quality.filter_threshold` |

---

<div align="center">
<sub>Built for TTS data collection · single-speaker purity over quantity 🎯</sub>
</div>
