from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(target)
    else:
        last_error: Exception | None = None
        frame = None
        for encoding in CSV_ENCODINGS:
            try:
                frame = pd.read_csv(target, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if frame is None:
            raise last_error or RuntimeError(f"Unable to read table: {target}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value
