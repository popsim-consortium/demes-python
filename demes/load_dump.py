"""
Functions to load and dump graphs in YAML and JSON formats.
"""
from __future__ import annotations
import contextlib
import json
import io
import math
from typing import Any, Generator, MutableMapping

import ruamel.yaml

import demes


@contextlib.contextmanager
def _open_file_polymorph(polymorph, mode="r"):
    """
    Open polymorph as a path and yield the fileobj. If that fails,
    just yield polymorph under the assumption it's a fileobj.
    """
    try:
        # We must specify utf8 explicitly for Windows.
        f = open(polymorph, mode, encoding="utf-8")
    except TypeError:
        f = polymorph
    try:
        yield f
    finally:
        if f is not polymorph:
            f.close()


# NOTE: The state of Python YAML libraries in 2020 leaves much to be desired.
# The pyyaml library supports only YAML v1.1, which has some awkward corner
# cases that have been fixed in YAML v1.2. A fork of pyaml, ruamel.yaml,
# does support YAML v1.2, and introduces a new API for parsing/emitting
# with additional features and desirable behaviour.
# However, neither pyyaml nor ruamel guarantee API stability, and neither
# provide complete reference documentation for their APIs.
# The YAML code in demes is limited to the following two functions,
# which are hopefully simple enough to not suffer from API instability.


def _load_yaml_asdict(fp) -> MutableMapping[str, Any]:
    with ruamel.yaml.YAML(typ="safe") as yaml:
        return yaml.load(fp)


def _dump_yaml_fromdict(data, fp, multidoc=False) -> None:
    """
    Dump data dict to a YAML file-like object.

    :param bool multidoc: If True, output the YAML document start line ``---``,
        and document end line ``...``, which indicate the beginning and end of
        a YAML document respectively. The start indicator is needed when
        outputting multiple YAML documents to a single file (or file stream).
        The end indicator is not strictly needed, but may be desirable
        depending on the underlying communication channel.
    """
    with ruamel.yaml.YAML(typ="safe", output=fp) as yaml:
        # Output flow style, but only for collections that consist only
        # of scalars (i.e. the leaves in the document tree).
        yaml.default_flow_style = None
        # Don't emit obscure unicode, output "\Uxxxxxxxx" instead.
        # Needed for string equality after round-tripping.
        yaml.allow_unicode = False
        # Keep dict insertion order, thank you very much!
        yaml.sort_base_mapping_type_on_output = False
        if multidoc:
            yaml.explicit_start = True
            yaml.explicit_end = True
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
    for default in data.get("defaults", []):
        if default in ["migration", "deme"]:
            start_time = data["defaults"][default].get("start_time")
            if start_time == _INFINITY_STR:
                data["defaults"][default]["start_time"] = float(start_time)


def _no_null_values(data: MutableMapping[str, Any]) -> None:
    """
    Checks for any null values in the input data.
    """

    def check_if_None(key, val):
        if val is None:
            raise ValueError(f"{key} must have a non-null value")

    def assert_no_nulls(d):
        for k, v in d.items():
            if isinstance(v, dict):
                assert_no_nulls(v)
            elif isinstance(v, list):
                for e in v:
                    if isinstance(e, dict):
                        assert_no_nulls(e)
                    else:
                        check_if_None(k, e)
            else:
                check_if_None(k, v)

    # Don't look inside metadata.
    data_no_metadata = {k: v for k, v in data.items() if k != "metadata"}
    assert_no_nulls(data_no_metadata)


def loads_asdict(string, *, format="yaml") -> MutableMapping[str, Any]:
    """
    Load a YAML or JSON string into a dictionary of nested objects.

    The input is *not* resolved, and is *not* validated.
    The returned object may be converted into a :class:`Builder`
    using :meth:`Builder.fromdict` or converted into a :class:`Graph`
    using :meth:`Graph.fromdict`.

    :param str string: The string to be loaded.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input string.
    :rtype: dict
    """
    with io.StringIO(string) as stream:
        return load_asdict(stream, format=format)


def load_asdict(filename, *, format="yaml") -> MutableMapping[str, Any]:
    """
    Load a YAML or JSON file into a dictionary of nested objects.

    The input is *not* resolved, and is *not* validated.
    The returned object may be converted into a :class:`Builder`
    using :meth:`Builder.fromdict` or converted into a :class:`Graph`
    using :meth:`Graph.fromdict`.

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
    elif format == "yaml":
        with _open_file_polymorph(filename) as f:
            data = _load_yaml_asdict(f)
    else:
        raise ValueError(f"unknown format: {format}")
    # We forbid null values in the input data.
    # See https://github.com/popsim-consortium/demes-spec/issues/76
    _no_null_values(data)
    # The string "Infinity" should only be present in JSON files.
    # But YAML is a superset of JSON, so we want the YAML loader to also
    # load JSON files without problem.
    _unstringify_infinities(data)
    return data


def loads(string, *, format="yaml") -> demes.Graph:
    """
    Load a graph from a YAML or JSON string.
    The keywords and structure of the input are defined by the
    :ref:`spec:sec_spec`.

    :param str string: The string to be loaded.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A resolved and validated demographic model.
    :rtype: demes.Graph
    """
    data = loads_asdict(string, format=format)
    return demes.Graph.fromdict(data)


def load(filename, *, format="yaml") -> demes.Graph:
    """
    Load a graph from a YAML or JSON file.
    The keywords and structure of the input are defined by the
    :ref:`spec:sec_spec`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a ``read()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
    :param str format: The format of the input file. Either "yaml" or "json".
    :return: A resolved and validated demographic model.
    :rtype: demes.Graph
    """
    data = load_asdict(filename, format=format)
    return demes.Graph.fromdict(data)


def load_all(filename) -> Generator[demes.Graph, None, None]:
    """
    Generate graphs from a YAML document stream. Documents must be separated by
    the YAML document start indicator, ``---``.
    The keywords and structure of each document are defined by the
    :ref:`spec:sec_spec`.

    :param filename: The path to the file to be loaded, or a file-like object
        with a ``read()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
    :return: A generator of resolved and validated demographic models.
    :rtype: Generator[demes.Graph, None, None]
    """
    with _open_file_polymorph(filename) as f:
        with ruamel.yaml.YAML(typ="safe") as yaml:
            for data in yaml.load_all(f):
                # We forbid null values in the input data.
                # See https://github.com/popsim-consortium/demes-spec/issues/76
                _no_null_values(data)
                # The string "Infinity" should only be present in JSON files.
                # But YAML is a superset of JSON, so we want the YAML loader to also
                # load JSON files without problem.
                _unstringify_infinities(data)
                yield demes.Graph.fromdict(data)


def dumps(graph, *, format="yaml", simplified=True) -> str:
    """
    Dump the specified graph to a YAML or JSON string.
    The keywords and structure of the output are defined by the
    :ref:`spec:sec_spec`.

    :param demes.Graph graph: The graph to dump.
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified:
        If True, returns a string following the :ref:`spec:sec_spec_hdm`,
        which has many fields omitted and is thus more compact.
        If False, returns a string that is fully-resolved following the
        :ref:`spec:sec_spec_mdm`.
    :return: The YAML or JSON string.
    :rtype: str
    """
    with io.StringIO() as stream:
        dump(graph, stream, format=format, simplified=simplified)
        string = stream.getvalue()
    return string


def dump(graph, filename, *, format="yaml", simplified=True) -> None:
    """
    Dump the specified graph to a file.
    The keywords and structure of the output are defined by the
    :ref:`spec:sec_spec`.

    :param demes.Graph graph: The graph to dump.
    :param filename: Path to the output file, or a file-like object with a
        ``write()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified:
        If True, the output file follows the :ref:`spec:sec_spec_hdm`,
        which has many fields omitted and is thus more compact.
        If False, the output file is fully-resolved and follows the
        :ref:`spec:sec_spec_mdm`.
    """
    if simplified:
        data = graph.asdict_simplified()
    else:
        data = graph.asdict()

    if format == "json":
        with _open_file_polymorph(filename, "w") as f:
            _stringify_infinities(data)
            json.dump(data, f, allow_nan=False, indent=2)
    elif format == "yaml":
        with _open_file_polymorph(filename, "w") as f:
            _dump_yaml_fromdict(data, f)
    else:
        raise ValueError(f"unknown format: {format}")


def dump_all(graphs, filename, *, simplified=True) -> None:
    """
    Dump the specified graphs to a multi-document YAML file or output stream.

    :param graphs: An iterable of graphs to dump.
    :param filename: Path to the output file, or a file-like object with a
        ``write()`` method.
    :type filename: Union[str, os.PathLike, FileLike]
    :param bool simplified:
        If True, the output file follows the :ref:`spec:sec_spec_hdm`,
        which has many fields omitted and is thus more compact.
        If False, the output file is fully-resolved and follows the
        :ref:`spec:sec_spec_mdm`.
    """
    with _open_file_polymorph(filename, "w") as f:
        for graph in graphs:
            if simplified:
                data = graph.asdict_simplified()
            else:
                data = graph.asdict()
            _dump_yaml_fromdict(data, f, multidoc=True)
