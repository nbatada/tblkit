from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tblkit")
except PackageNotFoundError:  # local dev
    __version__ = "0.0.0.dev0"

from . import core, utils
from .utils import UtilsAPI

__all__ = ["UtilsAPI", "core", "utils", "__version__"]
