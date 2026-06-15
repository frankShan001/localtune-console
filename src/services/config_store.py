"""Thread-safe, atomic project configuration storage."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import yaml

T = TypeVar("T")


class ProjectConfigStore:
    """Serialize configuration updates and never expose a partial YAML file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self) -> dict:
        with self.lock:
            return self._read_unlocked()

    def write(self, config: dict) -> None:
        with self.lock:
            self._write_unlocked(config)

    def update(self, mutator: Callable[[dict], T]) -> T:
        with self.lock:
            config = self._read_unlocked()
            result = mutator(config)
            self._write_unlocked(config)
            return result

    @contextmanager
    def transaction(self) -> Iterator[dict]:
        with self.lock:
            config = self._read_unlocked()
            yield config
            self._write_unlocked(config)

    def _read_unlocked(self) -> dict:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _write_unlocked(self, config: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()
