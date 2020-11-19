"""
Functions to load and dump graphs in YAML and JSON formats.
"""
import json

import strictyaml

import demes
from .schema import deme_graph_schema


def loads(string, *, format="yaml"):
    """
    Load a deme graph from a YAML or JSON string.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param str string: The string to be loaded.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A deme graph.
    :rtype: .DemeGraph
    """
    if format == "json":
        d = json.loads(string)
    elif format == "yaml":
        yaml = strictyaml.dirty_load(
            string, schema=deme_graph_schema, allow_flow_style=True
        )
        d = yaml.data
    else:
        raise ValueError(f"unknown format: {format}")
    return demes.DemeGraph.fromdict(d)


def load(filename, *, format="yaml"):
    """
    Load a deme graph from a YAML or JSON file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param filename: The path to the file to be loaded.
    :type filename: str or :class:`os.PathLike`
    :param str format: The format of the input file. Either "yaml" or "json".
    :return: A deme graph.
    :rtype: .DemeGraph
    """
    with open(filename) as f:
        return loads(f.read(), format=format)


def dumps(deme_graph, *, format="yaml", compact=True):
    """
    Dump the specified deme graph to a YAML or JSON string.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param .DemeGraph deme_graph: The deme graph to dump.
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool compact: If ``True``, a compact representation of the graph
        will be returned, where default and implicit values are removed.
        If ``False``, the complete graph will be returned.
    :return: The YAML or JSON string.
    :rtype: str
    """
    if compact:
        d = deme_graph.asdict_compact()
    else:
        d = deme_graph.asdict()

    if format == "json":
        string = json.dumps(d, indent=4)
    elif format == "yaml":
        doc = strictyaml.as_document(d, schema=deme_graph_schema)
        string = doc.as_yaml()
    else:
        raise ValueError(f"unknown format: {format}")

    return string


def dump(deme_graph, filename, *, format="yaml", compact=True):
    """
    Dump the specified deme graph to a file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param .DemeGraph deme_graph: The deme graph to dump.
    :param filename: Path to the output file.
    :type filename: str or :class:`os.PathLike`
    :param str format: The format of the output file. Either "yaml" or "json".
    """
    with open(filename, "w") as f:
        f.write(dumps(deme_graph, format=format, compact=compact))
