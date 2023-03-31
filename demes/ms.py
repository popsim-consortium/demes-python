from __future__ import annotations
import copy
import math
import argparse
import logging
import sys
import operator
import itertools
from typing import Any, List, Mapping, Set, Tuple

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
        _, exc, _ = sys.exc_info()
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


def float_str(a: float) -> str:
    """
    Convert float to string, for use in command line arguments.
    """
    if a < 0:
        # Use lots of decimal places for negative numbers because argparse
        # has problems parsing option args that are negative numbers in
        # exponential form.
        # https://github.com/popsim-consortium/demes-python/issues/325
        # https://bugs.python.org/issue9334
        return format(a, ".10f")
    else:
        return str(a)


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

    def __str__(self):
        s = ["-I", str(self.npop)] + [str(n) for n in self.n]
        if self.rate > 0:
            s += [float_str(self.rate)]
        return " ".join(s)


@attr.define
class Event(Option):
    # The first param of every demographic event is the event time.
    # For ms options without a time parameter (i.e. those that set the
    # initial simulation state), t is just set to zero.
    t = attr.ib(converter=float, validator=non_negative)


@attr.define
class GrowthRateChange(Event):
    # -G alpha
    # -eG t alpha
    alpha = attr.ib(converter=float, validator=finite)

    def __str__(self):
        if self.t > 0:
            s = ["-eG", float_str(self.t)]
        else:
            s = ["-G"]
        s.append(float_str(self.alpha))
        return " ".join(s)


@attr.define
class PopulationGrowthRateChange(Event):
    # -g i alpha
    # -eg t i alpha
    i = attr.ib(converter=int, validator=positive)
    alpha = attr.ib(converter=float, validator=finite)

    def __str__(self):
        if self.t > 0:
            s = ["-eg", float_str(self.t)]
        else:
            s = ["-g"]
        s.extend([str(self.i), float_str(self.alpha)])
        return " ".join(s)


@attr.define
class SizeChange(Event):
    # -eN t x
    x = attr.ib(converter=float, validator=non_negative)

    def __str__(self):
        return " ".join(["-eN", float_str(self.t), float_str(self.x)])


@attr.define
class PopulationSizeChange(Event):
    # -n i x
    # -en t i x
    i = attr.ib(converter=int, validator=positive)
    x = attr.ib(converter=float, validator=non_negative)

    def __str__(self):
        if self.t > 0:
            s = ["-en", float_str(self.t)]
        else:
            s = ["-n"]
        s.extend([str(self.i), float_str(self.x)])
        return " ".join(s)


@attr.define
class MigrationRateChange(Event):
    # -eM t x
    x = attr.ib(converter=float, validator=non_negative)

    def __str__(self):
        return " ".join(["-eM", float_str(self.t), float_str(self.x)])


@attr.define
class MigrationMatrixEntryChange(Event):
    # -m i j rate
    # -em t i j rate
    i = attr.ib(converter=int, validator=positive)
    j = attr.ib(converter=int, validator=positive)
    rate = attr.ib(converter=float, validator=non_negative)

    def __str__(self):
        if self.t > 0:
            s = ["-em", float_str(self.t)]
        else:
            s = ["-m"]
        s.extend([str(self.i), str(self.j), float_str(self.rate)])
        return " ".join(s)


@attr.define
class MigrationMatrixChange(Event):
    # -ma M11 M12 M13 ... M21 ...
    # -ema t npop M11 M12 M13 ... M21 ...
    npop = attr.ib(converter=int, validator=positive)
    mm_vector = attr.ib()

    @property
    def M(self):
        """
        Convert the args vector into a square list-of-lists matrix.
        """
        if len(self.mm_vector) != self.npop**2:
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

    def __str__(self):
        if self.t > 0:
            s = ["-ema", float_str(self.t)]
        else:
            s = ["-ma"]
        s.append(str(self.npop))
        M = self.M
        for j in range(self.npop):
            for k in range(self.npop):
                if j == k:
                    s.append("x")
                else:
                    s.append(float_str(M[j][k]))
        return " ".join(s)


@attr.define
class Split(Event):
    # -es t i p
    i = attr.ib(converter=int, validator=positive)
    p = attr.ib(converter=float, validator=unit_interval)

    def __str__(self):
        return " ".join(["-es", float_str(self.t), str(self.i), float_str(self.p)])


@attr.define
class Join(Event):
    # -ej t i j
    i = attr.ib(converter=int, validator=positive)
    j = attr.ib(converter=int, validator=positive)

    def __str__(self):
        return " ".join(["-ej", float_str(self.t), str(self.i), str(self.j)])


def build_parser(parser=None):
    if parser is None:
        parser = ValueErrorArgumentParser()

    class LoadFromFile(argparse.Action):
        def __call__(self, parser, namespace, filename, option_string=None):
            # parse arguments in the file and store them in the target namespace
            with open(filename) as f:
                args = f.read().split()
            parser.parse_args(args, namespace)

    parser.add_argument(
        "-f",
        metavar="filename",
        action=LoadFromFile,
        help="Insert commands from a file at this point in the command line.",
    )
    parser.add_argument(
        "-I",
        dest="structure",
        nargs="+",
        action=coerce_nargs(Structure.from_nargs),
        metavar=("num_demes", "n1"),
        help=(
            "Set the number of demes and the sampling configuration. "
            "The arguments are of the form "
            "'num_demes n1 n2 ... [4N0m]', "
            "specifying the number of demes, "
            "the sample configuration, and optionally, the migration "
            "rate for a symmetric island model (in units of 4 * N0). "
            "While values must be provided for the sample configuration, "
            "they are not used for constructing the Demes model."
        ),
    )
    parser.add_argument(
        "-n",
        nargs=2,
        action=coerce_nargs(lambda *x: PopulationSizeChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar=("i", "x"),
        help="Set the size of deme i to x * N0.",
    )
    parser.add_argument(
        "-g",
        nargs=2,
        action=coerce_nargs(lambda *x: PopulationGrowthRateChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar=("i", "alpha"),
        help="Set the growth rate of deme i to alpha.",
    )
    parser.add_argument(
        "-G",
        nargs=1,
        action=coerce_nargs(lambda *x: GrowthRateChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar="alpha",
        help="Set the growth rate to alpha for all demes.",
    )
    parser.add_argument(
        "-m",
        nargs=3,
        action=coerce_nargs(lambda *x: MigrationMatrixEntryChange(0, *x), append=True),
        dest="initial_state",
        default=[],
        metavar=("i", "j", "rate"),
        help=(
            "Sets an entry M[i, j] in the migration matrix to the "
            "specified rate. i and j are (1-indexed) deme IDs."
        ),
    )
    parser.add_argument(
        "-ma",
        nargs="+",
        action=coerce_nargs(
            lambda *x: MigrationMatrixChange.from_nargs(0, 1, *x), append=True
        ),
        dest="initial_state",
        default=[],
        metavar="entry",
        help=(
            "Sets the migration matrix from the specified vector of values. "
            "The entries are in the order M[1, 1], M[1, 2], ..., M[2, 1], "
            "M[2, 2], ..., M[N, N], where N is the number of demes. "
            "Diagonal entries may be written as 'x'."
        ),
    )
    parser.add_argument(
        "-eG",
        nargs=2,
        action=coerce_nargs(GrowthRateChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "alpha"),
        help="Set the growth rate for all demes to alpha at time t.",
    )
    parser.add_argument(
        "-eg",
        nargs=3,
        action=coerce_nargs(PopulationGrowthRateChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "alpha"),
        help="Set the growth rate of deme i to alpha at time t.",
    )
    parser.add_argument(
        "-eN",
        nargs=2,
        action=coerce_nargs(SizeChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "x"),
        help="Set the size of all demes to x * N0 at time t.",
    )
    parser.add_argument(
        "-en",
        nargs=3,
        action=coerce_nargs(PopulationSizeChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "x"),
        help="Set the size of deme i to x * N0 at time t.",
    )
    parser.add_argument(
        "-eM",
        nargs=2,
        action=coerce_nargs(MigrationRateChange, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "x"),
        help=(
            "Set the symmetric island model migration rate to "
            "'x / (num_demes - 1)' at time t."
        ),
    )
    parser.add_argument(
        "-em",
        action=coerce_nargs(MigrationMatrixEntryChange, append=True),
        dest="demographic_events",
        metavar=("t", "i", "j", "rate"),
        nargs=4,
        default=[],
        help=(
            "Sets the entry M[i, j] in the migration matrix to the "
            "specified rate at time t. i and j are (1-indexed) deme IDs."
        ),
    )
    parser.add_argument(
        "-ema",
        nargs="+",
        default=[],
        action=coerce_nargs(MigrationMatrixChange.from_nargs, append=True),
        dest="demographic_events",
        metavar=("t", "entry"),
        help=(
            "Sets the migration matrix from the specified vector of values "
            "at time t. "
            "The entries are in the order M[1, 1], M[1, 2], ..., M[2, 1], "
            "M[2, 2], ..., M[N, N], where N is the number of demes. "
            "Diagonal entries may be written as 'x'."
        ),
    )
    parser.add_argument(
        "-es",
        nargs=3,
        action=coerce_nargs(Split, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "p"),
        help=(
            "Split deme i into a new deme, such that the specified "
            "proportion p of lineages remains in deme i. The new deme has ID "
            "num_demes + 1, and has size N0, growth rate 0, and migration "
            "rates to and from the new deme are set to 0. "
            "Forwards in time this corresponds to an admixture event with "
            "the extinction of the new deme."
        ),
    )
    parser.add_argument(
        "-ej",
        nargs=3,
        action=coerce_nargs(Join, append=True),
        dest="demographic_events",
        default=[],
        metavar=("t", "i", "j"),
        help=(
            "Move all lineages in deme i to j at time t. All migration "
            "rates for deme i are set to zero. "
            "Forwards in time, this corresponds to a branch event "
            "in which lineages in j split into i."
        ),
    )
    return parser


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

    # Sort demographic events args by the time field.
    args.demographic_events.sort(key=operator.attrgetter("t"))
    # Process the initial_state options followed by the demographic_events.
    for t, events_iter in itertools.groupby(
        args.initial_state + args.demographic_events, operator.attrgetter("t")
    ):
        time = 4 * N0 * t
        events_group = list(events_iter)

        # Lineage movements matrix to track -es/ej (Split/Join) events.
        # This is used to turn complex sequences of -es/-ej events with the
        # same time parameter into more direct ancestry relationships.
        n = num_demes + sum(1 for event in events_group if isinstance(event, Split))
        lineage_movements = [[0] * n for _ in range(n)]
        for j in range(num_demes):
            lineage_movements[j][j] = 1
        # The params gleaned from Split/Join events, which are used to collapse
        # ancestry into plain old pulse migrations.
        split_join_params: List[Tuple[int, int, float]] = []

        for event in events_group:
            if isinstance(event, GrowthRateChange):
                # -G α
                # -eG t α
                growth_rate = event.alpha / (4 * N0)
                for j, deme in enumerate(b.data["demes"]):
                    if j not in joined:
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
                for j, deme in enumerate(b.data["demes"]):
                    if j not in joined:
                        current_epoch = deme["epochs"][0]
                        current_growth_rate = current_epoch.get("growth_rate", 0)
                        if (
                            current_growth_rate != 0
                            or current_epoch["end_size"] != size
                        ):
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

            elif isinstance(event, MigrationRateChange):
                # -eM t x
                mm = migration_matrix_at(time)
                for j in range(len(mm)):
                    if j not in joined:
                        for k in range(len(mm)):
                            if j != k and k not in joined:
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

            elif isinstance(event, Join):
                # -ej t i j
                # Move all lineages from deme i to deme j at time t.
                pop_i = convert_population_id(event.i)
                pop_j = convert_population_id(event.j)

                b.data["demes"][pop_i]["start_time"] = time
                b.data["demes"][pop_i]["ancestors"] = [f"deme{pop_j + 1}"]
                for lm in lineage_movements:
                    lm[pop_j] += lm[pop_i]
                    lm[pop_i] = 0
                for idx, (g, h, q) in reversed(list(enumerate(split_join_params))):
                    if h == pop_i:
                        split_join_params[idx] = (g, pop_j, q)
                        break
                else:
                    split_join_params.append((pop_i, pop_j, 1))

                mm = migration_matrix_at(time)
                # Turn off migrations to/from deme i.
                for k in range(num_demes):
                    if k != pop_i:
                        mm[k][pop_i] = 0
                        mm[pop_i][k] = 0

                # Record pop_i so that this index isn't used by later events.
                joined.add(pop_i)

            elif isinstance(event, Split):
                # -es t i p
                # Split deme i into a new deme (num_demes + 1),
                # with proportion p of lineages remaining in deme i,
                # and 1-p moving to the new deme.
                pid = convert_population_id(event.i)

                # Add new deme.
                new_pid = num_demes
                b.add_deme(
                    f"deme{new_pid + 1}",
                    start_time=math.inf,
                    epochs=[dict(end_size=N0, end_time=time)],
                )
                for lm in lineage_movements:
                    assert lm[new_pid] == 0
                    lm[new_pid] = (1 - event.p) * lm[pid]
                    lm[pid] *= event.p
                split_join_params.append((pid, new_pid, 1 - event.p))

                num_demes += 1

                # Expand each migration matrix with a row and column of zeros.
                for migration_matrix in mm_list:
                    for row in migration_matrix:
                        row.append(0)
                    migration_matrix.append([0 for _ in range(num_demes)])

            else:
                assert False, f"unhandled option: {event}"

        for j, k, p in split_join_params:
            ancestors = []
            proportions = []
            for o, proportion in enumerate(lineage_movements[j]):
                if j != o and proportion > 0:
                    ancestors.append(o)
                    proportions.append(proportion)
            if len(ancestors) == 0:
                continue
            p_jj = lineage_movements[j][j]
            if p_jj == 0:
                # No ancestry left in j.
                b.data["demes"][j]["ancestors"] = [f"deme{o + 1}" for o in ancestors]
                b.data["demes"][j]["proportions"] = proportions
            else:
                # Some ancestry is retained in j, so we use pulse migrations to
                # indicate foreign ancestry.
                # The order of pulses will later be reversed such that realised
                # ancestry proportions are maintained forwards in time.
                b.add_pulse(
                    sources=[f"deme{k + 1}"],
                    dest=f"deme{j + 1}",
                    time=time,
                    proportions=[p],
                )

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

    # Convert migration matrices into migration dictionaries.
    b._add_migrations_from_matrices(mm_list, mm_end_times)

    # Rescale rates so they don't have units of 4*N0.
    for migration in b.data["migrations"]:
        migration["rate"] /= 4 * N0

    # Remove demes whose existence time span is zero.
    # These can be created by simultaneous -es/-ej commands.
    b._remove_transient_demes()

    # Sort demes by their start time so that ancestors come before descendants.
    b._sort_demes_by_ancestry()

    # Reverse the order of pulses so realised ancestry proportions are correct.
    b.data.get("pulses", []).reverse()

    graph = b.resolve()
    return graph


def remap_deme_names(graph: demes.Graph, names: Mapping[str, str]) -> demes.Graph:
    assert sorted(names.keys()) == sorted(deme.name for deme in graph.demes)
    graph = graph.rename_demes(names)
    return graph


def from_ms(
    command: str,
    *,
    N0: float,
    deme_names: List[str] | None = None,
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

        \\text{deme size (diploid individuals)} &= N_0 x

        \\text{migration rate (per generation)} &= \\frac{M}{4 N_0}

    .. seealso::
        The :ref:`sec_cli_ms` command line interface.

    .. warning::

        Several programs exist that implement an ms or ms-like
        command line interface. But discrepancies do exist
        between ms and its clones for some command line arguments,
        and such discrepancies alter the interpretation of the
        demographic model.
        This function implements the behaviour as described in
        Hudson's ms manual. Command lines constructed for use with
        other programs may work as desired, but users are cautioned to
        carefully check the appropriate documentation to identify where
        discrepancies may exist.

        The ms manual may be obtained from
        http://home.uchicago.edu/~rhudson1/source/mksamples.html

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
    parser = build_parser()
    args, unknown = parser.parse_known_args(command.split())
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


def to_ms(graph: demes.Graph, *, N0, samples=None) -> str:
    """
    Get ms command line arguments for the graph.

    The order of deme IDs matches the order of demes in the graph.

    .. seealso::
        The ``--ms`` option of the :ref:`sec_cli_parse` command line interface.

    :param float N0:
        The reference population size used to translate into coalescent units.
        See :func:`from_ms` for details.
    :param list(int) samples:
        A list of haploid sample numbers, one for each deme, in the same
        order as the order of demes in the graph.
        The parameter is ignored for demographic models with only one deme.
        The given sampling scheme will be used with the ``-I`` argument.
        If not specified, the sampling configuration in the returned string
        will need editing prior to use.
    :return: The ms command line.
    :rtype: str
    """
    graph = graph.in_generations()
    cmd = []
    num_demes = len(graph.demes)
    if samples is not None and len(samples) != num_demes:
        raise ValueError("samples must match the number of demes in the graph")
    if num_demes > 1:
        if samples is None:
            # Output a no-samples configuration. The user must edit this anyway,
            # so if they blindly pass this to a simulator, it should complain.
            samples = [0] * num_demes
        structure = Structure.from_nargs(num_demes, *samples)
        cmd.append(str(structure))

    def get_growth_rate(epoch):
        ret = 0
        if epoch.size_function not in ["constant", "exponential"]:
            raise ValueError(
                "ms only supports constant or exponentially changing population sizes"
            )
        if epoch.end_size != epoch.start_size:
            dt = epoch.time_span / (4 * N0)
            ret = -math.log(epoch.start_size / epoch.end_size) / dt
        return ret

    events: List[Event] = []
    for j, deme in enumerate(graph.demes, 1):
        size = N0
        growth_rate = 0
        for epoch in reversed(deme.epochs):
            if size != epoch.end_size:
                size = epoch.end_size
                events.append(PopulationSizeChange(epoch.end_time, j, size / N0))
            alpha = get_growth_rate(epoch)
            if growth_rate != alpha:
                growth_rate = alpha
                events.append(
                    PopulationGrowthRateChange(epoch.end_time, j, growth_rate)
                )
            size = epoch.start_size

    # Describing either the ancestry of a deme with multiple ancestors,
    # or ancestry via a pulse migration, both require the use of -es/-ej to
    # first split the deme (or pulse dest) into two and then immediately join
    # one lineage to the ancestor (or pulse source). A split event creates
    # a new deme with ID equal to the current number of demes plus 1,
    # and we must know this ID in order to then join the new deme.
    # To obtain correct IDs, we sort the combined list of demes and pulses
    # by the deme start time or alternately the pulse time. IDs are then
    # sequential with this ordering.
    # Pulses are added in reverse order, so that pulses with the same time
    # are given the correct backwards-time ordering.
    # Pulses occurring at the same time as a deme's start time are possible,
    # and in this case the pulse will come first (backwards in time).
    demes_and_pulses = sorted(
        tuple(reversed(graph.pulses)) + tuple(graph.demes),
        key=lambda d_p: d_p.start_time if isinstance(d_p, demes.Deme) else d_p.time,
    )
    deme_id = {deme.name: j for j, deme in enumerate(graph.demes, 1)}

    for deme_or_pulse in demes_and_pulses:
        if isinstance(deme_or_pulse, demes.Deme):
            deme = deme_or_pulse
            for k, ancestor in enumerate(deme.ancestors):
                anc_id = deme_id[ancestor]
                proportion = deme.proportions[k] / sum(deme.proportions[k:])
                if k == len(deme.ancestors) - 1:
                    assert math.isclose(proportion, 1)
                    events.append(Join(deme.start_time, deme_id[deme.name], anc_id))
                else:
                    num_demes += 1
                    new_deme_id = num_demes
                    e1 = Split(deme.start_time, deme_id[deme.name], 1 - proportion)
                    e2 = Join(deme.start_time, new_deme_id, anc_id)
                    events.extend([e1, e2])
        else:
            assert isinstance(deme_or_pulse, demes.Pulse)
            pulse = deme_or_pulse
            num_demes += 1
            new_deme_id = num_demes
            if len(pulse.sources) > 1:
                raise ValueError(
                    "Currently pulses with only a single source are supported"
                )
            e1 = Split(pulse.time, deme_id[pulse.dest], 1 - pulse.proportions[0])
            e2 = Join(pulse.time, new_deme_id, deme_id[pulse.sources[0]])
            events.extend([e1, e2])

    # Turn migrations off at the start_time. We schedule all start_time
    # events first, and then all end_time events. This ensures that events
    # for migrations with an end_time that coincides with the start_time of
    # another migration will be scheduled later (backwards in time),
    # and thus override the rate=0 setting.
    for migration in graph.migrations:
        if (
            not math.isinf(migration.start_time)
            and migration.start_time != graph[migration.dest].start_time
            and migration.start_time != graph[migration.source].start_time
        ):
            events.append(
                MigrationMatrixEntryChange(
                    t=migration.start_time,
                    i=deme_id[migration.dest],
                    j=deme_id[migration.source],
                    rate=0,
                )
            )
    for migration in graph.migrations:
        events.append(
            MigrationMatrixEntryChange(
                t=migration.end_time,
                i=deme_id[migration.dest],
                j=deme_id[migration.source],
                rate=4 * N0 * migration.rate,
            )
        )

    events.sort(key=operator.attrgetter("t"))
    for event in events:
        event.t /= 4 * N0

    cmd.extend(str(event) for event in events)
    return " ".join(cmd)
