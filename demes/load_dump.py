"""
Functions to load and dump graphs in YAML and JSON formats.
"""
import json
import copy

import ruamel.yaml

import demes


def _loads_yaml_asdict(string):
    with ruamel.yaml.YAML(typ="safe", pure=True) as yaml:
        data = yaml.load(string)
    # split comma separated fields
    for deme in data.get("demes", []):
        if "ancestors" in deme:
            deme["ancestors"] = [anc.strip() for anc in deme["ancestors"].split(",")]
        if "proportions" in deme:
            deme["proportions"] = [float(p) for p in deme["proportions"].split(",")]
    for mig in data.get("migrations", {}).get("symmetric", []):
        mig["demes"] = [deme.strip() for deme in mig["demes"].split(",")]
    return data


def _dumps_yaml_fromdict(data):
    data = copy.deepcopy(data)
    # collapse comma separated fields
    for deme in data.get("demes", []):
        if "ancestors" in deme:
            deme["ancestors"] = ", ".join(deme["ancestors"])
        if "proportions" in deme:
            deme["proportions"] = ", ".join([str(p) for p in deme["proportions"]])
    for mig in data.get("migrations", {}).get("symmetric", []):
        mig["demes"] = ", ".join(mig["demes"])
    string = ruamel.yaml.dump(data)
    return string


def loads_asdict(string, *, format="yaml"):
    """
    Load a YAML or JSON string into a dictionary of nested objects.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param str string: The string to be loaded.
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input string.
    :rtype: dict
    """
    if format == "json":
        data = json.loads(string)
    elif format == "yaml":
        data = _loads_yaml_asdict(string)
    else:
        raise ValueError(f"unknown format: {format}")
    return data


def load_asdict(filename, *, format="yaml"):
    """
    Load a YAML or JSON file into a dictionary of nested objects.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param filename: The path to the file to be loaded.
    :type filename: str or :class:`os.PathLike`
    :param str format: The format of the input string. Either "yaml" or "json".
    :return: A dictionary of nested objects, with the same data model as the
        YAML or JSON input string.
    :rtype: dict
    """
    with open(filename) as f:
        return loads_asdict(f.read(), format=format)


def loads(string, *, format="yaml"):
    """
    Load a graph from a YAML or JSON string.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

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
    :ref:`schema <sec_schema>`.

    :param filename: The path to the file to be loaded.
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
    :ref:`schema <sec_schema>`.

    :param .Graph graph: The graph to dump.
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified: If True, returns a simplified graph. If False, returns
        a complete redundant graph.
    :return: The YAML or JSON string.
    :rtype: str
    """
    if simplified:
        data = graph.asdict_simplified()
    else:
        data = graph.asdict()

    if format == "json":
        string = json.dumps(data)
    elif format == "yaml":
        string = _dumps_yaml_fromdict(data)
    else:
        raise ValueError(f"unknown format: {format}")

    return string


def dump(graph, filename, *, format="yaml", simplified=True):
    """
    Dump the specified graph to a file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param .Graph graph: The graph to dump.
    :param filename: Path to the output file.
    :type filename: str or :class:`os.PathLike`
    :param str format: The format of the output file. Either "yaml" or "json".
    :param bool simplified: If True, outputs a simplified graph. If False, outputs
        a redundant graph.
    """
    with open(filename, "w") as f:
        f.write(dumps(graph, format=format, simplified=simplified))
