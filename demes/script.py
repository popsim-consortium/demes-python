"""
Functions to load and dump yaml formatted descriptions of a demography.
"""

import strictyaml

import demes
from .schema import deme_graph_schema


def loads(string):
    """
    Load a deme graph from a yaml-formatted string.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param str string: the ``yaml`` string to be loaded.
    :return: A deme graph.
    :rtype: .DemeGraph
    """
    yaml = strictyaml.dirty_load(
        string, schema=deme_graph_schema, allow_flow_style=True
    )
    d = yaml.data  # data dict
    g = demes.DemeGraph(
        description=d.get("description"),
        time_units=d.get("time_units"),
        generation_time=d.get("generation_time"),
        doi=d.get("doi"),
        selfing_rate=d.get("selfing_rate"),
        cloning_rate=d.get("cloning_rate"),
    )
    for deme_id, deme_dict in d.get("demes", dict()).items():
        if "epochs" in deme_dict:
            deme_dict["epochs"] = [
                demes.Epoch(**epoch_dict) for epoch_dict in deme_dict["epochs"]
            ]
        g.deme(deme_id, **deme_dict)
    for migration_type, migration_dict in d.get("migrations", dict()).items():
        if migration_type == "symmetric":
            for m in migration_dict:
                g.symmetric_migration(**m)
        if migration_type == "asymmetric":
            for m in migration_dict:
                g.migration(**m)
    for pulse_dict in d.get("pulses", []):
        g.pulse(**pulse_dict)
    # add population relationship events to the deme graph
    g.get_demographic_events()
    return g


def load(filename):
    """
    Load a deme graph from a ``yaml`` file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param filename: The path to the ``yaml`` file to load.
    :type filename: str or :class:`os.PathLike`
    :return: A deme graph.
    :rtype: .DemeGraph
    """
    with open(filename) as f:
        return loads(f.read())


def dumps(deme_graph):
    """
    Return a yaml-formatted string of the specified deme graph.
    The keywords and structure of the string are defined by the
    :ref:`schema <sec_schema>`.

    :param .DemeGraph deme_graph: The deme graph to dump.
    :return: A yaml-formatted string.
    :rtype: str
    """
    d = deme_graph.asdict_compact()
    doc = strictyaml.as_document(d, schema=deme_graph_schema)
    return doc.as_yaml()


def dump(deme_graph, filename):
    """
    Dump the specified deme graph to a ``yaml`` file.
    The keywords and structure of the file are defined by the
    :ref:`schema <sec_schema>`.

    :param .DemeGraph deme_graph: The deme graph to dump.
    :param filename: Path to the output file.
    :type filename: str or :class:`os.PathLike`
    """
    with open(filename, "w") as f:
        f.write(dumps(deme_graph))
