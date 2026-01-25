# ruff: noqa
"""Deprecated shim for ``src.processing.neural.lshn_network``.

Use ``src.cognition.processing.neural.lshn_network``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`src.processing.neural.lshn_network` is deprecated; use `src.cognition.processing.neural.lshn_network`.",
    DeprecationWarning,
    stacklevel=2,
)

from src.cognition.processing.neural.lshn_network import *  # type: ignore

# Some unit tests import this symbol explicitly. Star imports don't bring in
# underscore-prefixed names, so re-export it here for backward compatibility.
from src.cognition.processing.neural.lshn_network import _DummyHopfieldLayer  # type: ignore
