import copy
import math
import argparse
import logging
import sys
from typing import Any, Dict, List, Mapping, MutableMapping, Set, Tuple

import attr

import demes
from .demes import finite, positive, non_negative, unit_interval

logger = logging.getLogger(__name__)


class ValueErrorArgumentParser(argparse.ArgumentParser):
    """
    An AgumentParser that throws a ValueError instead of exiting.
    It would be preferred to catch and reraise TypeError and ValueError
    separately, but that isn't possible without touching ArgumentParser
    internals because those errors are caught by parse_known_args(),
    which then throws an argparse.ArgumentError.
    So we just use ValueError for everything.
    """

    def error(self, message):
        _, exc, traceback = sys.exc_info()
        raise ValueError(str(exc))


def coerce_nargs(obj_creator, append=False):
    """
    Return an argparse action that coerces the collected nargs into
    data class using the specified class/function ``obj_creator``.
    This simplifies the validation of heterogeneously-typed args lists.
    If ``append`` is True, the option may be specified multiple times
    and each coerced object will be appended to a list.

    .. code::

        @attr.define
        class Foo
            n = attr.ib(converter=int)
            x = attr.ib(converter=float)

        parser.add_argument(
            "--foo",
            nargs=2,
            action=coerce_nargs(Foo),
            ...
        )

        args = parser.parse_args("--foo 6 1e-5")
        assert isinstance(args.foo, Foo)

    """
    parent_class = argparse.Action
    if append:
        parent_class = argparse._AppendAction

    class CoerceAction(parent_class):
        def __call__(self, parser, namespace, values, *args, **kwargs):
            obj = obj_creator(*values)
            # We save the option strings to later identify which option was
            # used to create the object. E.g. -n and -en options both map to
            # PopulationSizeChange, but have slightly different semantics.
            obj.option_strings = self.option_strings
            if append:
                super().__call__(parser, namespace, obj, *args, **kwargs)
            else:
                setattr(namespace, self.dest, obj)

    return CoerceAction


##
# Data classes for ms options. These help to separate input validation
# from the graph building procedure.


@attr.define
class Option:
    # This attribute is set by CoerceAction.
    option_strings: List[str] = attr.ib(init=False)


@attr.define
class Structure(Option):
    # -I npop n1 n2 ... [4*N0*m]
    npop = attr.ib(converter=int, validator=positive)
    n: Any = attr.ib()  # samples list: currently ignored
    rate = attr.ib(converter=float, validator=non_negative)

    def __attrs_post_init__(self):
        if len(self.n) != self.npop:
            raise ValueError("sample configuration doesn't match number of demes")

    @classmethod
    def from_nargs(cls, *args):
        npop, *n = args
        rate = 0
        try:
            npop = int(npop)
        except ValueError:
            raise ValueError(f"-I 'npop' ({args[0]}) not an integer")
        if len(n) == npop + 1:
            *n, rate = n
        return cls(npop, n, rate)


@attr.define
class Event(Option):
    # The first param of every demographic event is the event time.
    # For ms options without a time parameter (i.e. those that set the
    # initial simulation state), t is just set to zero.
    t = attr.ib(converter=float, validator=non_negative)


@attr.define
class GrowthRateChange(Event):
    # -G α
    # -eG t α
    alpha = attr.ib(converter=float, validator=finite)


@attr.define
class PopulationGrowthRateChange(Event):
    # -g i α
    # -eg t i α
    i = attr.ib(converter=int, validator=positive)
    alpha = attr.ib(converter=float, validator=finite)


@attr.define
class SizeChange(Event):
    # -eN t x
    x = attr.ib(converter=float, validator=non_negative)


@attr.define
class PopulationSizeChange(Event):
    # -n i x
    # -en t i x
    i = attr.ib(converter=int, validator=positive)
    x = attr.ib(converter=float, validator=non_negative)


@attr.define
class MigrationRateChange(Event):
    # -eM t x
    x = attr.ib(converter=float, validator=non_negative)


@attr.define
class MigrationMatrixEntryChange(Event):
    # -m i j rate
    # -em t i j rate
    i = attr.ib(converter=int, validator=positive)
    j = attr.ib(converter=int, validator=positive)
    rate = attr.ib(converter=float, validator=non_negative)


@attr.define
class MigrationMatrixChange(Event):
    # -ma M11 M12 M12 ... M21 ...
    # -ema t npop M11 M12 M12 ... M21 ...
    npop = attr.ib(converter=int, validator=positive)
    mm_vector = attr.ib()

    @property
    def M(self):
        """
        Convert the args vector into a square list-of-lists matrix.
        """
        if len(self.mm_vector) != self.npop ** 2:
            raise ValueError(
                f"Must be npop^2={self.npop**2} migration matrix entries: "
                f"{self.mm_vector}"
            )
        migration_matrix = [[0 for j in range(self.npop)] for k in range(self.npop)]
        for j in range(self.npop):
            for k in range(self.npop):
                if j != k:
                    rate = self.mm_vector[j * self.npop + k]

                    # Convert to float if possible. Ms ignores migration matrix
                    # diagonals, as well as migration entries for "joined" demes.
                    # The manual suggests to indicate diagonal elements with:
                    #       x's, or any symbol one chooses to make the matrix
                    #       more readable.
                    # NaNs will be caught later during graph resolution if we
                    # really attempt to use the value.
                    try:
                        rate = float(rate)
                    except ValueError:
                        rate = math.nan

                    migration_matrix[j][k] = rate
        return migration_matrix

    @classmethod
    def from_nargs(cls, *args):
        t, npop, *mm_vector = args
        return cls(t, npop, mm_vector)


@attr.define
class Admixture(Event):
    # -es t i p
    i = attr.ib(converter=int, validator=positive)
    p = attr.ib(converter=float, validator=unit_interval)


@attr.define
class PopulationSplit(Event):
    # -ej t i j
    i = attr.ib(converter=int, validator=positive)
    j = attr.ib(converter=int, validator=positive)


def parse_ms_args(command: str):
    parser = ValueErrorArgumentParser()

    class LoadFromFile(argparse.Action):
        def __call__(self, parser, namespace, filename, option_string=None):
            # parse arguments in the file and store them in the target namespace
            with open(filename) as f:
                args = f.read().split()
            parser.parse_args(args, namespace)

    parser.add_argument(
        "-f",
        "--filename",
        action=LoadFromFile,
        help="Insert commands from a file at this point in the command line.",
    )
    parser.add_argument(
        "--structure",
        "-I",
        nargs="+",
        action=coerce_nargs(Structure.from_nargs),
        metavar="value",
        help=(
            "Sample from populations with the specified deme structure. "
            "The arguments are of the form 'num_demes "
            "n1 n2 ... [4N0m]', specifying the number of populations, "
            "the sample configuration, and optionally, the migration "
            "rate for a symmetric island model"
        ),
    )
    parser.add_argument(
        "--population-size",
        "-n",
        nargs=2,
        action=coerce_nargs(lambda *x: PopulationSizeChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar=("i", "x"),
        help="Set the size of a specific population to size*N0.",
    )
    parser.add_argument(
        "--population-growth-rate",
        "-g",
        nargs=2,
        action=coerce_nargs(lambda *x: PopulationGrowthRateChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar=("i", "α"),
        help="Set the growth rate to alpha for a specific population.",
    )
    parser.add_argument(
        "--growth-rate",
        "-G",
        nargs=1,
        action=coerce_nargs(lambda *x: GrowthRateChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar="α",
        help="Set the growth rate to alpha for all populations.",
    )
    parser.add_argument(
        "--migration-matrix-entry",
        "-m",
        nargs=3,
        action=coerce_nargs(lambda *x: MigrationMatrixEntryChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar=("i", "j", "rate"),
        help=(
            "Sets an entry M[i, j] in the migration matrix to the "
            "specified rate. i and j are (1-indexed) population "
            "IDs. Multiple options can be specified."
        ),
    )
    parser.add_argument(
        "--migration-matrix",
        "-ma",
        nargs="+",
        action=coerce_nargs(
            lambda *x: MigrationMatrixChange.from_nargs(0, 1, *x), append=True
        ),
        dest="initial_state",
        default=[],
        metavar="entry",
        help=(
            "Sets the migration matrix to the specified value. The "
            "entries are in the order M[1, 1], M[1, 2], ..., M[2, 1], "
            "M[2, 2], ..., M[N, N], where N is the number of populations. "
            "Diagonal entries may be written as 'x'."
        ),
    )
    parser.add_argument(
        "--growth-rate-change",
        "-eG",
        nargs=2,
        action=coerce_nargs(GrowthRateChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "alpha"),
        help="Set the growth rate for all populations to alpha at time t",
    )
    parser.add_argument(
        "--population-growth-rate-change",
        "-eg",
        nargs=3,
        action=coerce_nargs(PopulationGrowthRateChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "alpha"),
        help=("Set the growth rate for a specific population to " "alpha at time t"),
    )
    parser.add_argument(
        "--size-change",
        "-eN",
        nargs=2,
        action=coerce_nargs(SizeChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "x"),
        help="Set the population size for all populations to x * N0 at time t",
    )
    parser.add_argument(
        "--population-size-change",
        "-en",
        nargs=3,
        action=coerce_nargs(PopulationSizeChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "x"),
        help=(
            "Set the population size for a specific population to " "x * N0 at time t"
        ),
    )
    parser.add_argument(
        "--migration-rate-change",
        "-eM",
        nargs=2,
        action=coerce_nargs(MigrationRateChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "x"),
        help=(
            "Set the symmetric island model migration rate to "
            "x / (npop - 1) at time t"
        ),
    )
    parser.add_argument(
        "--migration-matrix-entry-change",
        "-em",
        action=coerce_nargs(MigrationMatrixEntryChange, append=True),
        dest="demographic_events",
        metavar=("time", "i", "j", "rate"),
        nargs=4,
        default=[],
        help=(
            "Sets an entry M[i, j] in the migration matrix to the "
            "specified rate at the specified time. i and j are "
            "(1-indexed) population IDs."
        ),
    )
    parser.add_argument(
        "--migration-matrix-change",
        "-ema",
        nargs="+",
        default=[],
        action=coerce_nargs(MigrationMatrixChange.from_nargs, append=True),
        dest="demographic_events",
        metavar="entry",
        help=(
            "Sets the migration matrix to the specified value at time t. "
            "The entries are in the order M[1, 1], M[1, 2], ..., M[2, 1], "
            "M[2, 2], ..., M[N, N], where N is the number of populations. "
            "Diagonal entries may be written as 'x'."
        ),
    )
    parser.add_argument(
        "--admixture",
        "-es",
        nargs=3,
        action=coerce_nargs(Admixture, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "p"),
        help=(
            "Split the specified population into a new population, such "
            "that the specified proportion of lineages remains in "
            "the population i. Forwards in time this "
            "corresponds to an admixture event. The new population has ID "
            "num_demes + 1. Migration rates to and from the new "
            "population are set to 0, and growth rate is 0 and the "
            "population size for the new population is N0."
        ),
    )
    parser.add_argument(
        "--population-split",
        "-ej",
        nargs=3,
        action=coerce_nargs(PopulationSplit, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "j"),
        help=(
            "Move all lineages in population i to j at time t. "
            "Forwards in time, this corresponds to a population split "
            "in which lineages in j split into i. All migration "
            "rates for population i are set to zero."
        ),
    )

    args, unknown = parser.parse_known_args(command.split())
    # Sort demographic events args by the time field.
    args.demographic_events.sort(key=lambda x: x.t)
    return args, unknown


def demes_sorted_by_ancestry(demes_data: List[Mapping]):
    """
    Return the demes list sorted so that ancestors come before descendants.
    """
    # Iterate through the remaining demes and insert them once all
    # ancestors are already inserted.
    remaining = demes_data.copy()
    inserted = dict()
    n_remaining = len(remaining)
    while n_remaining > 0:
        for deme in remaining.copy():
            ancestors_in_graph = True
            for ancestor in deme.get("ancestors", []):
                if ancestor not in inserted:
                    ancestors_in_graph = False
                    break
            if ancestors_in_graph:
                remaining.remove(deme)
                inserted[deme["name"]] = deme
        if n_remaining == len(remaining):
            raise ValueError(
                "Cannot resolve graph because an ancestor/descendant cycle "
                f"exists among: {deme['name'] for deme in remaining}"
            )
        n_remaining = len(remaining)
    return list(inserted.values())


def migrations_from_mm_list(
    mm_list: List[List[List[float]]], end_times: List[float], deme_names: List[str]
) -> List[MutableMapping]:
    """
    Convert a list of migration matrices into a list of migration dicts.
    """
    assert len(mm_list) == len(end_times)
    migrations: List[MutableMapping] = []
    current: Dict[Tuple[int, int], MutableMapping] = dict()
    start_time = math.inf
    for migration_matrix, end_time in zip(mm_list, end_times):
        n = len(migration_matrix)
        assert n == len(deme_names)
        for j in range(n):
            assert n == len(migration_matrix[j])
            for k in range(n):
                if j == k:
                    continue
                rate = migration_matrix[j][k]
                mm = current.get((j, k))
                if mm is None:
                    if rate != 0:
                        mm = dict(
                            source=deme_names[j],
                            dest=deme_names[k],
                            start_time=start_time,
                            end_time=end_time,
                            rate=rate,
                        )
                        current[(j, k)] = mm
                        migrations.append(mm)
                else:
                    if rate == 0:
                        del current[(j, k)]
                    elif mm["rate"] == rate:
                        # extend mm
                        mm["end_time"] = end_time
                    else:
                        mm = dict(
                            source=deme_names[j],
                            dest=deme_names[k],
                            start_time=start_time,
                            end_time=end_time,
                            rate=rate,
                        )
                        current[(j, k)] = mm
                        migrations.append(mm)
        start_time = end_time
    return migrations


def build_graph(args, N0: float) -> demes.Graph:
    num_demes = 1
    # List of migration matrices in time-descending order (oldest to most recent).
    mm_list = [[[0.0]]]
    # End times for each migration matrix.
    mm_end_times = [0.0]
    # Indexes of demes that have been joined (-ej option).
    joined: Set[int] = set()

    if args.structure is not None:
        # -I npop n1 n2 .. [rate]
        num_demes = args.structure.npop
        if num_demes > 1:
            mm_list[0] = [
                [
                    args.structure.rate / (num_demes - 1) * int(j != k)
                    for j in range(num_demes)
                ]
                for k in range(num_demes)
            ]

    # Start building the demography.
    b = demes.Builder()
    for j in range(num_demes):
        initial_epoch = dict(end_size=N0, end_time=0)
        b.add_deme(f"deme{j + 1}", start_time=math.inf, epochs=[initial_epoch])

    def convert_population_id(population_id):
        """
        Checks the specified population ID makes sense and returns
        it as a zero-based index into the demes list.
        """
        if population_id < 1 or population_id > num_demes:
            raise ValueError(
                f"Bad population ID '{population_id}': "
                f"must be between 1 and num_demes ({num_demes})"
            )
        pid = population_id - 1
        if pid in joined:
            raise ValueError(
                f"Bad population ID '{population_id}': "
                "population previously joined with -ej"
            )
        return pid

    def epoch_resolve(deme, time):
        """
        Return the oldest epoch if it has end_time == time. If not, create a
        new oldest epoch with end_time=time. Also resolve sizes by dealing
        with the growth_rate attribute (if required).
        """
        epoch = deme["epochs"][0]
        start_time = deme["start_time"]
        end_time = epoch["end_time"]
        if not (start_time > time >= end_time):
            raise ValueError(
                f"time {time} outside {deme['name']}'s existence interval "
                f"(start_time={start_time}, end_time={end_time}]"
            )

        if time > end_time:
            new_epoch = copy.deepcopy(epoch)
            # find size at given time
            growth_rate = epoch.pop("growth_rate", 0)
            dt = time - epoch["end_time"]
            size_at_t = epoch["end_size"] * math.exp(-growth_rate * dt)
            epoch["start_size"] = size_at_t
            new_epoch["end_size"] = size_at_t
            new_epoch["end_time"] = time
            deme["epochs"].insert(0, new_epoch)
            epoch = new_epoch

        return epoch

    def migration_matrix_at(time):
        """
        Return the oldest migration matrix if it has end_time == time. If not,
        create a new oldest migration matrix with end_time = time.
        """
        migration_matrix = mm_list[0]
        if time > mm_end_times[0]:
            # We need a new migration matrix.
            migration_matrix = copy.deepcopy(migration_matrix)
            mm_list.insert(0, migration_matrix)
            mm_end_times.insert(0, time)
        return migration_matrix

    # Process the initial_state options followed by the demographic_events.
    for event in args.initial_state + args.demographic_events:
        time = 4 * N0 * event.t
        if isinstance(event, GrowthRateChange):
            # -G α
            # -eG t α
            growth_rate = event.alpha / (4 * N0)
            for deme in b.data["demes"]:
                current_epoch = deme["epochs"][0]
                current_growth_rate = current_epoch.get("growth_rate", 0)
                if current_growth_rate != growth_rate:
                    epoch = epoch_resolve(deme, time)
                    epoch["growth_rate"] = growth_rate

        elif isinstance(event, PopulationGrowthRateChange):
            # -g i α
            # -eg t i α
            pid = convert_population_id(event.i)
            growth_rate = event.alpha / (4 * N0)
            deme = b.data["demes"][pid]
            current_epoch = deme["epochs"][0]
            current_growth_rate = current_epoch.get("growth_rate", 0)
            if current_growth_rate != growth_rate:
                epoch = epoch_resolve(deme, time)
                epoch["growth_rate"] = growth_rate

        elif isinstance(event, SizeChange):
            # -eN t x
            size = event.x * N0
            for deme in b.data["demes"]:
                current_epoch = deme["epochs"][0]
                current_growth_rate = current_epoch.get("growth_rate", 0)
                if current_growth_rate != 0 or current_epoch["end_size"] != size:
                    epoch = epoch_resolve(deme, time)
                    epoch["growth_rate"] = 0
                    epoch["end_size"] = size

        elif isinstance(event, PopulationSizeChange):
            # -n i x
            # -en t i x
            pid = convert_population_id(event.i)
            size = event.x * N0
            deme = b.data["demes"][pid]
            current_epoch = deme["epochs"][0]
            current_growth_rate = current_epoch.get("growth_rate", 0)
            if current_growth_rate != 0 or current_epoch["end_size"] != size:
                epoch = epoch_resolve(deme, time)
                epoch["end_size"] = size
                # set growth_rate to 0 for -en option, but not for -n option
                if "-en" in event.option_strings:
                    epoch["growth_rate"] = 0

        elif isinstance(event, PopulationSplit):
            # -ej t i j
            pop_i = convert_population_id(event.i)
            pop_j = convert_population_id(event.j)

            b.data["demes"][pop_i]["start_time"] = time
            b.data["demes"][pop_i]["ancestors"] = [f"deme{pop_j + 1}"]

            mm = migration_matrix_at(time)
            # Turn off migrations to/from deme i.
            for k in range(num_demes):
                if k != pop_i:
                    mm[k][pop_i] = 0
                    mm[pop_i][k] = 0

            # Record pop_i so that this index isn't used by later events.
            joined.add(pop_i)

        elif isinstance(event, Admixture):
            # -es t i p
            pid = convert_population_id(event.i)

            # Add a new deme which will be the source of a migration pulse.
            new_pid = num_demes
            b.add_deme(
                f"deme{new_pid + 1}",
                start_time=math.inf,
                epochs=[dict(end_size=N0, end_time=time)],
            )
            # In ms, the probability of staying in source is p and the
            # probabilty of moving to the new population is 1 - p.
            b.add_pulse(
                source=f"deme{new_pid + 1}",
                dest=f"deme{pid + 1}",
                time=time,
                proportion=1 - event.p,
            )
            num_demes += 1

            # Expand each migration matrix with a row and column of zeros.
            for migration_matrix in mm_list:
                for row in migration_matrix:
                    row.append(0)
                migration_matrix.append([0 for _ in range(num_demes)])

        ##
        # Demographic events that affect the migration matrix

        elif isinstance(event, MigrationRateChange):
            # -eM t x
            mm = migration_matrix_at(time)
            for j in range(len(mm)):
                for k in range(len(mm)):
                    if j != k:
                        mm[j][k] = event.x / (num_demes - 1)

        elif isinstance(event, MigrationMatrixEntryChange):
            # -m i j x
            # -em t i j x
            pid_i = convert_population_id(event.i)
            pid_j = convert_population_id(event.j)
            if pid_i == pid_j:
                raise ValueError("Cannot set diagonal elements in migration matrix")
            mm = migration_matrix_at(time)
            mm[pid_i][pid_j] = event.rate

        elif isinstance(event, MigrationMatrixChange):
            # -ma M11 M12 M12 ... M21 ...
            # -ema t npop M11 M12 M12 ... M21 ...
            if "-ma" in event.option_strings:
                event.npop = num_demes
            if event.npop != num_demes:
                raise ValueError(
                    f"-ema 'npop' ({event.npop}) doesn't match the current "
                    f"number of demes ({num_demes})"
                )
            _ = migration_matrix_at(time)
            mm = mm_list[0] = copy.deepcopy(event.M)
            # Ms ignores matrix entries for demes that were previously joined
            # (-ej option), and users may deliberately put invalid values
            # here (e.g. 'x'). So we explicitly set these rates to zero.
            for j in joined:
                for k in range(num_demes):
                    if j != k:
                        mm[j][k] = 0
                        mm[k][j] = 0
        else:
            assert False, f"unhandled option: {event}"

    # Resolve/remove growth_rate in oldest epochs.
    for deme in b.data["demes"]:
        start_time = deme.get("start_time", math.inf)
        epoch = deme["epochs"][0]
        growth_rate = epoch.pop("growth_rate", 0)
        if growth_rate != 0:
            if math.isinf(start_time):
                raise ValueError(
                    f"{deme['name']}: growth rate for infinite-length epoch is invalid"
                )
            dt = start_time - epoch["end_time"]
            epoch["start_size"] = epoch["end_size"] * math.exp(-dt * growth_rate)
        else:
            epoch["start_size"] = epoch["end_size"]

    migrations = migrations_from_mm_list(
        mm_list, mm_end_times, [deme["name"] for deme in b.data["demes"]]
    )
    # Rescale rates so they don't have units of 4*N0.
    for migration in migrations:
        migration["rate"] /= 4 * N0
    b.data["migrations"] = migrations

    # Reinsert each deme so that ancestors come before descendants.
    b.data["demes"] = demes_sorted_by_ancestry(b.data["demes"])

    graph = b.resolve()
    return graph


def remap_deme_names(graph: demes.Graph, names: Mapping[str, str]) -> demes.Graph:
    assert sorted(names.keys()) == sorted(deme.name for deme in graph.demes)
    graph = copy.deepcopy(graph)
    for deme in graph.demes:
        deme.name = names[deme.name]
        deme.ancestors = [names[ancestor] for ancestor in deme.ancestors]
    for migration in graph.migrations:
        migration.source = names[migration.source]
        migration.dest = names[migration.dest]
    for pulse in graph.pulses:
        pulse.source = names[pulse.source]
        pulse.dest = names[pulse.dest]
    for k, deme in list(graph._deme_map.items()):
        del graph._deme_map[k]
        graph._deme_map[names[k]] = deme
    return graph


def from_ms(
    command: str,
    *,
    N0: float,
    deme_names: List[str] = None,
) -> demes.Graph:
    """
    Convert an ms demographic model into a demes graph.

    `Hudson's ms <https://doi.org/10.1093/bioinformatics/18.2.337>`_
    uses coalescent units for times (:math:`t`),
    population sizes (:math:`x`), and migration rates (:math:`M`).
    These will be converted to more familiar units using the given
    ``N0`` value (:math:`N_0`) according to the following rules:

    .. math::

        \\text{time (in generations)} &= 4 N_0 t

        \\text{deme size (haploid individuals)} &= N_0 x

        \\text{migration rate (per generation)} &= \\frac{M}{4 N_0}

    :param str command: The ms command line.
    :param float N0:
        The reference population size (:math:`N_0`) used to translate
        from coalescent units. For a ``command`` that specifies a
        :math:`\\theta` value with the ``-t theta`` option,
        this can be calculated as :math:`N_0 = \\theta / (4 \\mu L)`,
        where :math:`\\mu` is the per-generation mutation rate and
        :math:`L` is the length of the sequence being simulated.
    :param list[str] deme_names: A list of names to use for the demes.
        If not specified, demes will be named deme1, deme2, etc.

    :return: The demes graph.
    :rtype: demes.Graph
    """
    args, unknown = parse_ms_args(command)
    if len(unknown) > 0:
        # TODO: do something better here? Pass unknown args back to user?
        logger.warning(f"Ignoring unknown args: {unknown}")

    graph = build_graph(args, N0)
    if deme_names is not None:
        if len(set(deme_names)) != len(graph.demes):
            raise ValueError(
                f"graph has {len(graph.demes)} unique demes, "
                f"but deme_names has {len(set(deme_names))}"
            )
        name_map = dict(zip((f"deme{j+1}" for j in range(len(deme_names))), deme_names))
        graph = remap_deme_names(graph, name_map)
    return graph
