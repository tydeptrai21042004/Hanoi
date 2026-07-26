"""Compatibility module for the independent SP-SCQIM DCT-Schur proposal.

The former duplicate implementations were split: the inactive historical code
remains in :mod:`direct_schur_rescue_legacy`, while this public module exposes
only the validated API of :mod:`schur_coupling_qim`.
"""
from .schur_coupling_qim import *  # noqa: F401,F403
from .schur_coupling_qim import __all__
