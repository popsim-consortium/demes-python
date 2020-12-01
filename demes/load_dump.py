"""
Functions to load and dump graphs in YAML and JSON formats.
"""
import json
import contextlib

import ruamel.yaml

import demes


@contextlib.contextmanager
def _open_whatever(path_or_fileobj, mode="r"):
    """
    Open path_or_fileobj as a path and yield the fileobj. If that fails,
    just yield path_or_fileobj under the assumption it's a fileobj.
    """
    try:
        with open(path_or_fileobj, mode) as f:
            yield f
    except TypeError:
        yield path_or_fileobj


def _load_yaml_asdict(fp):
    with ruamel.yaml.YAML(typ="safe", pure=True) as yaml:
        return yaml.load(fp)


def _dump_yaml_fromdict(data, fp):
    with ruamel.yaml.YAML(typ="safe", pure=True, output=fp) as yaml:
        yaml.default_flow_style = False  # don't output json arrays/objects
        yaml.dump(data)


def load_asdict(filename, *, format="yaml"):
    """
    Load a YAML or JSON file into a dictionary of nested objects.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a `read()` method.
    :type filename: str or :class:`os.PathLike` or file object.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input string.
    :rtype: dict
    """
    if format == "json":
        with _open_whatever(filename) as f:
            data = json.load(f)
    elif format == "yaml":
        with _open_whatever(filename) as f:
            data = _load_yaml_asdict(f)
    else:
        raise ValueError(f"unknown format: {format}")
    return data


def load(filename, *, format="yaml"):
    """
    Load a graph from a YAML or JSON file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a `read()` method.
    :type filename: str or :class:`os.PathLike` or file object.
    :param str format: The format of the input file. Either "yaml" or "json".
    :return: A graph.
    :rtype: .Graph
    """
    data = load_asdict(filename, format=format)
    return demes.Graph.fromdict(data)


def dump(graph, filename, *, format="yaml", simplified=True):
    """
    Dump the specified graph to a file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param .Graph graph: The graph to dump.
    :param filename: Path to the output file, or a file-like object with a
        `write()` method.
    :type filename: str or :class:`os.PathLike` or file object.
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified: If True, outputs a simplified graph. If False, outputs
        a redundant graph.
    """
    if simplified:
        data = graph.asdict_simplified()
    else:
        data = graph.asdict()

    if format == "json":
        with _open_whatever(filename, "w") as f:
            json.dump(data, f)
    elif format == "yaml":
        with _open_whatever(filename, "w") as f:
            _dump_yaml_fromdict(data, f)
    else:
        raise ValueError(f"unknown format: {format}")
