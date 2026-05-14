"""Lightweight file helpers (package expects this module when importing src.utils)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


class FileUtils:
    """Minimal file utilities used by the utils package namespace."""

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def safe_filename(name: str, default: str = "file") -> str:
        base = os.path.basename(str(name)) or default
        return base.replace("\x00", "")
