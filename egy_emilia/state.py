"""Tiny resumable-state helper. Each stage records completed items in a JSON file
under state_dir, so re-running skips finished work and continues where it stopped."""

import json
from pathlib import Path


class Checkpoint:
    """A set of 'done' keys persisted to disk. Call .done(key) to check,
    .mark(key) to record. Cheap, append-style, safe to interrupt."""

    def __init__(self, state_dir: Path, name: str):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / f"{name}.json"
        self._done: set[str] = set()
        if self.path.exists():
            try:
                self._done = set(json.loads(self.path.read_text()))
            except Exception:
                self._done = set()

    def done(self, key: str) -> bool:
        return key in self._done

    def mark(self, key: str) -> None:
        self._done.add(key)
        self.path.write_text(json.dumps(sorted(self._done), ensure_ascii=False, indent=0))

    def __len__(self) -> int:
        return len(self._done)
