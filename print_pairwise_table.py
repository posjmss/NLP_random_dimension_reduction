"""Run the package console table helper directly from the repository root."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
# Add src to sys.path so `python print_pairwise_table.py ...` works directly.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random_embedding_truncation.print_pairwise_table import main


if __name__ == "__main__":
    main()
