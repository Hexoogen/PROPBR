"""Run the synthetic workflow directly from a checkout after dependencies are installed."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from propbr.cli import main

if __name__ == "__main__":
    main(["demo", *sys.argv[1:]])
