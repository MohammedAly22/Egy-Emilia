"""EGY-Emilia · entry point — Stage 1: download audios.

Edit config.yaml (paths.sources_file, download.*), then:  python download.py
"""

from egy_emilia.config import load_config
from egy_emilia.download import run

if __name__ == "__main__":
    run(load_config())
