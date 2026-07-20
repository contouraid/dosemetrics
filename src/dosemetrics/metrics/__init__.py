"""Radiotherapy dose metrics organized by domain and comparison semantics.

Single-plan quantities are computed in their clinical domain modules:
``dvh``, ``conformity``, and ``homogeneity``. Reference-based plan metrics
live in ``comparison`` and consistently accept ``reference`` before
``evaluated``. Lower-level image comparisons remain in ``dose_comparison``.

The package intentionally exports modules rather than a flat collection of
functions. This keeps similarly named single-plan indices and between-plan
distances distinct at every call site.
"""

from . import comparison
from . import conformity
from . import dose_comparison
from . import dvh
from . import gamma
from . import geometric
from . import homogeneity

__all__ = [
    "comparison",
    "conformity",
    "dose_comparison",
    "dvh",
    "gamma",
    "geometric",
    "homogeneity",
]
