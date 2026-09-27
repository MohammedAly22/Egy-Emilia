"""EGY-Emilia · entry point — run a SINGLE stage.

Set STAGE below to one of: download, diarize, loudness, quality, transcribe, publish.
Then:  python run_stage.py
"""

from egy_emilia import diarize_stage, download, loudness, publish, quality, transcribe
from egy_emilia.config import load_config

# ---- choose which stage to run -------------------------------------------- #
STAGE = "transcribe"
# --------------------------------------------------------------------------- #

_STAGES = {
    "download": download.run,
    "diarize": diarize_stage.run,
    "loudness": loudness.run,
    "quality": quality.run,
    "transcribe": transcribe.run,
    "publish": publish.run,
}

if __name__ == "__main__":
    _STAGES[STAGE](load_config())
