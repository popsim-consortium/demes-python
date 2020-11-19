# flake8: noqa: F401

__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

from .demes import Epoch, Migration, Pulse, Deme, Graph, Split, Branch, Merge, Admix
from .load_dump import load, loads, dump, dumps
