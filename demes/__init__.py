__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

from .demes import (
    Builder,
    Epoch,
    AsymmetricMigration,
    Pulse,
    Deme,
    Graph,
    Split,
    Branch,
    Merge,
    Admix,
)
from .load_dump import (
    load_asdict,
    loads_asdict,
    load,
    loads,
    load_all,
    dump,
    dumps,
    dump_all,
)
from .ms import from_ms, to_ms

__all__ = [
    "Builder",
    "Epoch",
    "AsymmetricMigration",
    "Pulse",
    "Deme",
    "Graph",
    "Split",
    "Branch",
    "Merge",
    "Admix",
    "load_asdict",
    "loads_asdict",
    "load",
    "loads",
    "load_all",
    "dump",
    "dumps",
    "dump_all",
    "from_ms",
    "to_ms",
]


# Override the symbols that are returned when calling dir(<module-name>).
# https://www.python.org/dev/peps/pep-0562/
# We do this because the Python REPL and IPython notebooks ignore __all__
# when providing autocomplete suggestions. They instead rely on dir().
# By not showing internal symbols in the dir() output, we reduce the chance
# that users rely on non-public features.
def __dir__():
    return sorted(__all__)
