"""EGY-Emilia · entry point — run the FULL pipeline end-to-end.

  download → diarize+chunk → loudness → quality → transcribe → publish (HF, private)

Every stage is resumable; re-running only does what's left.
Edit config.yaml, then:  python run_pipeline.py
"""
from egy_emilia.pipeline import run_all

if __name__ == "__main__":
    run_all()
