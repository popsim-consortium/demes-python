"""
Functions to load and dump graphs in YAML and JSON formats.
"""
import contextlib
import json
import io
import math
from typing import MutableMapping, Any

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
        # Output flow style, but only for collections that consist only
        # of scalars (i.e. the leaves in the document tree).
        yaml.default_flow_style = None
        # Don't emit obscure unicode, output "\Uxxxxxxxx" instead.
        # Needed for string equality after round-tripping.
        yaml.allow_unicode = False
        # Keep dict insertion order, thank you very much!
        yaml.sort_base_mapping_type_on_output = False
        yaml.dump(data)


_INFINITY_STR = "Infinity"


def _stringify_infinities(data: MutableMapping[str, Any]) -> None:
    """
    Modifies the data dict so infinite values are set to the string "Infinity".

    This is done for JSON output, because the JSON spec explicitly does not
    provide an encoding for infinity.  The string-valued "Infinity" is likely
    to be parsed without problem, and is converted to a float inf by the
    following implementations:
     * Python: float("Infinity")
     * JavaScript: parseFloat("Infinity")
     * C: strtod("Infinity", NULL) and atof("Infinity")
     * R: as.numeric("Infinity")
     * SLiM/Eidos: asFloat("Infinity")
    """
    for deme in data["demes"]:
        if "start_time" in deme and math.isinf(deme["start_time"]):
            deme["start_time"] = _INFINITY_STR
    for migration in data.get("migrations", []):
        if "start_time" in migration and math.isinf(migration["start_time"]):
            migration["start_time"] = _INFINITY_STR


def _unstringify_infinities(data: MutableMapping[str, Any]) -> None:
    """
    Modifies the data dict so the string "Infinity" is converted to float.
    """
    for deme in data["demes"]:
        start_time = deme.get("start_time")
        if start_time == _INFINITY_STR:
            deme["start_time"] = float(start_time)
    for migration in data.get("migrations", []):
        start_time = migration.get("start_time")
        if start_time == _INFINITY_STR:
            migration["start_time"] = float(start_time)


def loads_asdict(string, *, format="yaml"):
    """
    Load a YAML or JSON string into a dictionary of nested objects.
    The keywords and structure of the input are defined by the
    :ref:`spec:sec_ref`.

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
    The keywords and structure of the input are defined by the
    :ref:`spec:sec_ref`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a ``read()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
    :param str format: The format of the input file. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input.
    :rtype: dict
    """
    if format == "json":
        with _open_file_polymorph(filename) as f:
            data = json.load(f)
            _unstringify_infinities(data)
    elif format == "yaml":
        with _open_file_polymorph(filename) as f:
            data = _load_yaml_asdict(f)
    else:
        raise ValueError(f"unknown format: {format}")
    return data


def loads(string, *, format="yaml"):
    """
    Load a graph from a YAML or JSON string.
    The keywords and structure of the input are defined by the
    :ref:`spec:sec_ref`.

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
    The keywords and structure of the input are defined by the
    :ref:`spec:sec_ref`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a ``read()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
    :param str format: The format of the input file. Either "yaml" or "json".
    :return: A graph.
    :rtype: .Graph
    """
    data = load_asdict(filename, format=format)
    return demes.Graph.fromdict(data)


def dumps(graph, *, format="yaml", simplified=True):
    """
    Dump the specified graph to a YAML or JSON string.
    The keywords and structure of the output are defined by the
    :ref:`spec:sec_ref`.

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
    The keywords and structure of the output are defined by the
    :ref:`spec:sec_ref`.

    :param .Graph graph: The graph to dump.
    :param filename: Path to the output file, or a file-like object with a
        ``write()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
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
            _stringify_infinities(data)
            json.dump(data, f, allow_nan=False)
    elif format == "yaml":
        with _open_file_polymorph(filename, "w") as f:
            _dump_yaml_fromdict(data, f)
    else:
        raise ValueError(f"unknown format: {format}")
