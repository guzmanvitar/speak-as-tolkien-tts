"""Defines project constants"""

from pathlib import Path

this = Path(__file__)

ROOT = this.parents[1]

LOGS = ROOT / "logs"

SRC = ROOT / "src"

MODELS = SRC / "models"

SECRETS = ROOT / ".secrets"

DATA = ROOT / "data"
DATA_RECORDINGS = DATA / "recordings"
DATA_PROCESSED = DATA / "processed"

TESTS_DIR = ROOT / "tests"

LOGS.mkdir(exist_ok=True, parents=True)
DATA_RECORDINGS.mkdir(exist_ok=True, parents=True)
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True)
