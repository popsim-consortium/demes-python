"""
Functions to load and dump yaml formatted descriptions of a demography.
The strictyaml schema defined here follows the DemeGraph construction API.
"""
# TODO: add symmetric_migration and subgraph schemas.

from strictyaml import (
    Optional,
    Map,
    MapPattern,
    Float,
    Int,
    Seq,
    Str,
    dirty_load,
    as_document,
)

import demes

Number = Int() | Float()

_epoch_schema = Map(
    {
        "start_time": Number,
        Optional("end_time"): Number,
        Optional("initial_size"): Number,
        Optional("final_size"): Number,
    }
)

_migration_schema = Map(
    {
        Optional("start_time"): Number,
        Optional("end_time"): Number,
        "source": Str(),
        "dest": Str(),
        "rate": Float(),
    }
)

_pulse_schema = Map(
    {"time": Number, "source": Str(), "dest": Str(), "proportion": Float()}
)

_deme_schema = Map(
    {
        Optional("ancestor"): Str(),
        Optional("start_time"): Number,
        Optional("end_time"): Number,
        Optional("initial_size"): Number,
        Optional("final_size"): Number,
        Optional("epochs"): Seq(_epoch_schema),
    }
)

_deme_graph_schema = Map(
    {
        "description": Str(),
        "time_units": Str(),
        "generation_time": Number,
        Optional("doi"): Str(),
        Optional("default_Ne"): Number,
        "demes": MapPattern(Str(), _deme_schema),
        Optional("migrations"): Seq(_migration_schema),
        Optional("pulses"): Seq(_pulse_schema),
    }
)


def loads(string):
    """
    Load a deme graph from a yaml-formatted string.

    :param str string: the ``yaml`` string to be loaded.
    :return: A deme graph.
    :rtype: .DemeGraph

    .. todo:: Describe demes' yaml format. The semantics in the yaml
        follow that of the :class:`.DemeGraph` demography construction
        methods.
    """
    yaml = dirty_load(string, schema=_deme_graph_schema, allow_flow_style=True)
    d = yaml.data  # data dict
    g = demes.DemeGraph(
        description=d.get("description"),
        time_units=d.get("time_units"),
        generation_time=d.get("generation_time"),
        doi=d.get("doi"),
        default_Ne=d.get("default_Ne"),
    )
    for deme_id, deme_dict in d.get("demes", dict()).items():
        if "epochs" in deme_dict:
            deme_dict["epochs"] = [
                demes.Epoch(**epoch_dict) for epoch_dict in deme_dict["epochs"]
            ]
        g.deme(deme_id, **deme_dict)
    for migration_dict in d.get("migrations", []):
        g.migration(**migration_dict)
    for pulse_dict in d.get("pulses", []):
        g.pulse(**pulse_dict)
    return g


def load(filename):
    """
    Load a deme graph from a yaml-formatted file.

    :param str filename: the name of the ``yaml`` file to load.
    :return: A deme graph.
    :rtype: .DemeGraph
    """
    with open(filename) as f:
        return loads(f.read())


def dumps(deme_graph):
    """
    Return a yaml-formatted string of the specified deme graph.

    :param .DemeGraph deme_graph: The deme graph to dump.
    :return: A yaml-formatted string.
    :rtype: str
    """
    d = deme_graph.asdict_compact()
    doc = as_document(d, schema=_deme_graph_schema)
    return doc.as_yaml()


def dump(deme_graph, filename):
    """
    Dump the specified deme graph to a yaml-formatted file.

    :param .DemeGraph deme_graph: The deme graph to dump.
    :param str filename: Name of the output file.
    """
    with open(filename, "w") as f:
        f.write(dumps(deme_graph))
