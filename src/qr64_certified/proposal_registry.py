"""Backward-compatible import for the proposal registry."""
from .proposals import proposal_registry as _implementation

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
