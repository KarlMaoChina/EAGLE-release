from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "eagle.egg-info"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".txt", ".csv", ".toml", ".cfg", ".ini"}
FORBIDDEN = (
    "maoshufan",
    "liutao",
    "/data/",
    "\\data\\maoshufan",
    "\\data\\liutao",
    "rn_",
    "xh0",
    "chict",
    "shufan",
    "nnunetverified",
    "rafeature",
    "concat_5",
    "train_4modal",
    "dual_spacing_enhance",
    "drop_names",
    "ly2025",
    "renji",
    "xinhua",
    "pacs",
    "有无",
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "test_leakage.py":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files.append(path)
    return files


def test_no_site_or_internal_identifiers() -> None:
    hits: list[str] = []
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in FORBIDDEN:
            if token in lowered:
                hits.append(f"{path.relative_to(ROOT)}: {token}")
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            hits.append(f"{path.relative_to(ROOT)}: cjk")
    assert hits == []
