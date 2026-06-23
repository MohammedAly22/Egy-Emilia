"""EGY-Emilia — end-to-end pipeline for gathering clean single-speaker TTS data."""

# Force audio backends to soundfile BEFORE torch/torchaudio/transformers import.
# torchcodec (the new default decoder) tries to dlopen libnvrtc.so.13 (CUDA 13),
# which doesn't exist on this CUDA-12 box and spams OSError tracebacks. Pinning
# the backend to soundfile sidesteps torchcodec entirely.
import os as _os

_os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")
_os.environ.setdefault("TORIO_USE_FFMPEG", "0")
# transformers: prefer soundfile/librosa over torchcodec for audio decoding
_os.environ.setdefault("TRANSFORMERS_NO_TORCHCODEC", "1")

__version__ = "0.1.0"
