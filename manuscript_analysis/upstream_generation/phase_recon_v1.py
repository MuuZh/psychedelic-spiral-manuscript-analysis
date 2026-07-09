#!/usr/bin/env python3
"""Compatibility entry point for the spiral phase-field reconstruction script.

The implementation now lives in:
    manuscript_analysis/spiral_phase_model/phase_recon_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "spiral_phase_model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from phase_recon_v1 import main  # noqa: E402


if __name__ == "__main__":
    main()
