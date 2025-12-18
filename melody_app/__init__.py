"""melody_app - Industrial-grade melody & harmony transcription pipeline.

Main entry:
  python -m melody_app.cli ...

This project is designed for practical, enterprise-style robustness:
- modular backends (CREPE / pYIN / BasicPitch / piano_transcription_inference)
- stable audio preprocessing
- polyphonic->monophonic melody extraction with DP (Viterbi)
- optional chord estimation (template baseline / Omnizart)
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "2.0.0"
