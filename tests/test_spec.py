"""
Check that we do the right thing with the demes-spec test cases,
by comparing against the reference implementation.
"""
import math
import json
import operator
import pathlib
import subprocess

import pytest

import demes


def get_test_cases(which_subset):
    """Return a list of test case files."""
    assert which_subset in ("valid", "invalid")
    cwd = pathlib.Path(__file__).parent.resolve()
    test_dir = cwd / ".." / "demes-spec" / "test-cases" / which_subset
    files = [str(file) for file in test_dir.glob("*.yaml")]
    assert len(files) > 1
    return files


def resolve_ref(filename) -> dict:
    """Resolve YAML file using the reference implementation."""
    cwd = pathlib.Path(__file__).parent.resolve()
    resolver = (
        cwd / ".." / "demes-spec" / "reference_implementation" / "resolve_yaml.py"
    )
    with subprocess.Popen(
        # "-X utf8" is needed on Windows.
        ["python3", "-X", "utf8", str(resolver), filename],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ) as p:
        assert p.stdout is not None  # pacify mypy
        stdout = p.stdout.read()
    assert p.returncode == 0
    return json.loads(stdout)


def dict_isclose(d1, d2, rel_tol=1e-9, abs_tol=1e-12):
    """Return True if nested dicts d1 and d2 are close, False otherwise."""
    if isinstance(d1, dict) and isinstance(d2, dict):
        return d1.keys() == d2.keys() and all(
            dict_isclose(d1[k], d2[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in d1
        )
    elif isinstance(d1, list) and isinstance(d2, list):
        return all(
            dict_isclose(l1, l2, rel_tol=rel_tol, abs_tol=abs_tol)
            for l1, l2 in zip(d1, d2)
        )
    elif isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        return math.isclose(d1, d2, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        return d1 == d2


def dict_map(f, d):
    """Apply function f to all terminal elements of dict d."""
    if isinstance(d, dict):
        return {k: dict_map(f, v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_map(f, x) for x in d]
    else:
        return f(d)


@pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
@pytest.mark.parametrize("filename", get_test_cases("valid"))
def test_valid(filename):
    d1 = demes.load(filename).asdict()
    d2 = resolve_ref(filename)

    # Normalise infinities so they're all floats.
    def cast_inf(x):
        math.inf if x == "Infinity" else x

    d1 = dict_map(cast_inf, d1)
    d2 = dict_map(cast_inf, d2)

    # The order of migrations could be different, depending on the order in
    # which symmetric migrations were resolved into pairs.
    migrations_key = operator.itemgetter("source", "dest", "end_time")
    d1["migrations"].sort(key=migrations_key)
    d2["migrations"].sort(key=migrations_key)

    if not dict_isclose(d1, d2):
        # Assert equality (which will fail) so that pytest prints a nice diff.
        assert d1 == d2


@pytest.mark.parametrize("filename", get_test_cases("invalid"))
def test_invalid(filename):
    with pytest.raises(Exception):
        demes.load(filename)
