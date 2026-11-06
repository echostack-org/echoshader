"""Compatibility alias for the basic viewer, now named single_viewer."""

import sys
from . import single_viewer

sys.modules[__name__] = single_viewer
