"""
Functions to load and dump graphs in YAML and JSON formats.
"""
import contextlib
import json
import io

import ruamel.yaml

import demes


@contextlib.contextmanager
def _open_file_polymorph(polymorph, mode="r"):
    """
    Open polymorph as a path and yield the fileobj. If that fails,
    just yield polymorph under the assumption it's a fileobj.
    """
    try:
        with open(polymorph, mode) as f:
            yield f
    except TypeError:
        yield polymorph


# NOTE: The state of Python YAML libraries in 2020 leaves much to be desired.
# The pyyaml library supports only YAML v1.1, which has some awkward corner
# cases that have been fixed in YAML v1.2. A fork of pyaml, ruamel.yaml,
# does support YAML v1.2, and introduces a new API for parsing/emitting
# with additional features and desirable behaviour.
# However, neither pyyaml nor ruamel gaurantee API stability, and neither
# provide complete reference documentation for their APIs.
# The YAML code in demes is limited to the following two functions,
# which are hopefully simple enough to not suffer from API instability.


def _load_yaml_asdict(fp):
    with ruamel.yaml.YAML(typ="safe") as yaml:
        return yaml.load(fp)


def _dump_yaml_fromdict(data, fp):
    with ruamel.yaml.YAML(typ="safe", output=fp) as yaml:
        # Disable JSON-style inline arrays and dicts.
        yaml.default_flow_style = False
        # Don't emit obscure unicode, output "\Uxxxxxxxx" instead.
        # Needed for string equality after round-tripping.
        yaml.allow_unicode = False
        # Keep dict insertion order, thank you very much!
        yaml.sort_base_mapping_type_on_output = False
        yaml.dump(data)


def loads_asdict(string, *, format="yaml"):
    """
    Load a YAML or JSON string into a dictionary of nested objects.
    The keywords and structure of the string are defined by the
    :ref:`spec:sec_spec`.

    :param str string: The string to be loaded.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input string.
    :rtype: dict
    """
    with io.StringIO(string) as stream:
        return load_asdict(stream, format=format)


def load_asdict(filename, *, format="yaml"):
    """
    Load a YAML or JSON file into a dictionary of nested objects.
    The keywords and structure of the string are defined by the
    :ref:`spec:sec_spec`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a `read()` method.
    :type filename: str or :class:`os.PathLike` or file object
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input string.
    :rtype: dict
    """
    if format == "json":
        with _open_file_polymorph(filename) as f:
            data = json.load(f)
    elif format == "yaml":
        with _open_file_polymorph(filename) as f:
            data = _load_yaml_asdict(f)
    else:
        raise ValueError(f"unknown format: {format}")
    return data


def loads(string, *, format="yaml"):
    """
    Load a graph from a YAML or JSON string.
    The keywords and structure of the string are defined by the
    :ref:`spec:sec_spec`.

    :param str string: The string to be loaded.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A graph.
    :rtype: .Graph
    """
    data = loads_asdict(string, format=format)
    return demes.Graph.fromdict(data)


def load(filename, *, format="yaml"):
    """
    Load a graph from a YAML or JSON file.
    The keywords and structure of the file are defined by the
    :ref:`spec:sec_spec`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a `read()` method.
    :type filename: str or :class:`os.PathLike`
    :param str format: The format of the input file. Either "yaml" or "json".
    :return: A graph.
    :rtype: .Graph
    """
    data = load_asdict(filename, format=format)
    return demes.Graph.fromdict(data)


def dumps(graph, *, format="yaml", simplified=True):
    """
    Dump the specified graph to a YAML or JSON string.
    The keywords and structure of the string are defined by the
    :ref:`spec:sec_spec`.

    :param .Graph graph: The graph to dump.
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified: If True, returns a simplified graph. If False, returns
        a complete redundant graph.
    :return: The YAML or JSON string.
    :rtype: str
    """
    with io.StringIO() as stream:
        dump(graph, stream, format=format, simplified=simplified)
        string = stream.getvalue()
    return string


def dump(graph, filename, *, format="yaml", simplified=True):
    """
    Dump the specified graph to a file.
    The keywords and structure of the file are defined by the
    :ref:`spec:sec_spec`.

    :param .Graph graph: The graph to dump.
    :param filename: Path to the output file, or a file-like object with a
        `write()` method.
    :type filename: str or :class:`os.PathLike` or file object
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified: If True, outputs a simplified graph. If False, outputs
        a redundant graph.
    """
    if simplified:
        data = graph.asdict_simplified()
    else:
        data = graph.asdict()

    if format == "json":
        with _open_file_polymorph(filename, "w") as f:
            json.dump(data, f)
    elif format == "yaml":
        with _open_file_polymorph(filename, "w") as f:
            _dump_yaml_fromdict(data, f)
    else:
        raise ValueError(f"unknown format: {format}")
