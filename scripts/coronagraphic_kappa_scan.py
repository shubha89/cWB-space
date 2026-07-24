"""Script entry point for pyTDI-A/E WDM selection and coronagraphic kappa scan."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHORO_DIR = PROJECT_ROOT / "choronagraphic-cluster"
if str(CHORO_DIR) not in sys.path:
    sys.path.insert(0, str(CHORO_DIR))

import run_one_smbhb as proto  # noqa: E402


def main() -> None:
    proto.main()


if __name__ == "__main__":
    main()
