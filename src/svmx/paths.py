"""
Project path constants derived from the repository layout.

Usage:
    from src.svmx.paths import PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR
"""

from pathlib import Path

# Repository root is two levels above this file: src/svmx/paths.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_SAMPLE = DATA_DIR / "sample"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"