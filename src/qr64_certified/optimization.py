"""Backward-compatible import for the proposal optimization utilities."""
from .proposals import optimization as _implementation

globals().update({
    name: value
    for name, value in vars(_implementation).items()
    if not name.startswith("__")
})

__all__ = getattr(
    _implementation,
    "__all__",
    [name for name in globals() if not name.startswith("_")],
)
