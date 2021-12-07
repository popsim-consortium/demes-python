import sys
import pathlib
import tempfile

import pytest
from demes import *


def test_builder():
    b = Builder()
    b.add_deme("A", epochs=[dict(start_size=100)])
    b.resolve()


def test_dumps_and_loads():
    b = Builder()
    b.add_deme("A", epochs=[dict(start_size=100)])
    graph1 = b.resolve()
    dump_str = dumps(graph1)
    graph2 = loads(dump_str)
    graph1.assert_close(graph2)


def test_dump_and_load():
    b = Builder()
    b.add_deme("A", epochs=[dict(start_size=100)])
    graph1 = b.resolve()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = pathlib.Path(tmpdir) / "temp.yaml"
        dump(graph1, tmpfile)
        graph2 = load(tmpfile)
    graph1.assert_close(graph2)


def test_public_symbols():
    Builder
    Epoch
    AsymmetricMigration
    Pulse
    Deme
    Graph
    Split
    Branch
    Merge
    Admix

    load_asdict
    loads_asdict
    load
    loads
    load_all
    dump
    dumps
    dump_all

    from_ms


def test_nonpublic_symbols():
    with pytest.raises(NameError):
        demes
    with pytest.raises(NameError):
        load_dump
    with pytest.raises(NameError):
        ms
    with pytest.raises(NameError):
        prec32


PY36 = sys.version_info[0:2] < (3, 7)


@pytest.mark.xfail(PY36, reason="__dir__ does nothing on Python 3.6", strict=True)
def test_demes_dir():
    import demes

    dir_demes = set(dir(demes))
    assert "load" in dir_demes
    assert "dump" in dir_demes
    assert "loads" in dir_demes
    assert "dumps" in dir_demes

    assert "demes" not in dir_demes
    assert "load_dump" not in dir_demes
    assert "ms" not in dir_demes
    assert "graphs" not in dir_demes
    assert "prec32" not in dir_demes
