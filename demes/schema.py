"""
The YAML subset accepted by ``demes`` is defined here as a ``strictyaml`` schema.
"""

from strictyaml import (
    CommaSeparated,
    Map,
    MapPattern,
    Float,
    Int,
    Optional,
    Seq,
    Str,
)

Number = Int() | Float()

epoch_schema = Map(
    {
        Optional("start_time"): Number,
        "end_time": Number,
        Optional("initial_size"): Number,
        Optional("final_size"): Number,
        Optional("size_function"): Str(),
        Optional("selfing_rate"): Number,
        Optional("cloning_rate"): Number,
    }
)

asymmetric_migration_schema = Map(
    {
        Optional("start_time"): Number,
        Optional("end_time"): Number,
        "source": Str(),
        "dest": Str(),
        "rate": Float(),
    }
)

symmetric_migration_schema = Map(
    {
        Optional("start_time"): Number,
        Optional("end_time"): Number,
        "demes": CommaSeparated(Str()),
        "rate": Float(),
    }
)

pulse_schema = Map(
    {"time": Number, "source": Str(), "dest": Str(), "proportion": Float()}
)

deme_schema = Map(
    {
        Optional("description"): Str(),
        Optional("ancestors"): CommaSeparated(Str()),
        Optional("proportions"): CommaSeparated(Float()),
        Optional("start_time"): Number,
        Optional("end_time"): Number,
        Optional("initial_size"): Number,
        Optional("final_size"): Number,
        Optional("epochs"): Seq(epoch_schema),
        Optional("selfing_rate"): Number,
        Optional("cloning_rate"): Number,
    }
)

deme_graph_schema = Map(
    {
        "description": Str(),
        "time_units": Str(),
        Optional("generation_time"): Number,
        Optional("doi"): Str(),
        Optional("default_Ne"): Number,
        "demes": MapPattern(Str(), deme_schema),
        Optional("migrations"): Map(
            {
                Optional("symmetric"): Seq(symmetric_migration_schema),
                Optional("asymmetric"): Seq(asymmetric_migration_schema),
            }
        ),
        Optional("pulses"): Seq(pulse_schema),
        Optional("selfing_rate"): Number,
        Optional("cloning_rate"): Number,
    }
)
