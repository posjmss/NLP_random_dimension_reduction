"""Add the vendored FinMTEB clone to sys.path so `finance_mteb` is importable.

FinMTEB is vendored at <repo_root>/third_party/FinMTEB and is not packaged on
PyPI, so we extend sys.path rather than installing it.
"""

import sys
from pathlib import Path


def add_finmteb_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    finmteb_path = repo_root / "third_party" / "FinMTEB"
    if not finmteb_path.exists():
        raise FileNotFoundError(
            f"Vendored FinMTEB not found at {finmteb_path}. "
            "Run: git clone https://github.com/yixuantt/FinMTEB.git "
            f"{finmteb_path}"
        )
    finmteb_str = str(finmteb_path)
    if finmteb_str not in sys.path:
        sys.path.insert(0, finmteb_str)
    return finmteb_path
