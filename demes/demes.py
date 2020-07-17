from typing import List, Mapping, Union
import itertools
import math
import collections

import attr

Number = Union[int, float]
ID = str
Time = Number
Size = Number
Rate = float
Proportion = float


# Validator functions.


def positive(self, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be greater than zero")


def non_negative(self, attribute, value):
    if value < 0:
        raise ValueError(f"{attribute.name} must be non-negative")


def finite(self, attribute, value):
    if math.isinf(value):
        raise ValueError(f"{attribute.name} must be finite")


def unit_interval(self, attribute, value):
    if not (0 <= value <= 1):
        raise ValueError(f"must have 0 <= {attribute.name} <= 1")


def optional(func):
    """
    Wraps one or more validator functions with an "if not None" clause.
    """
    if isinstance(func, (tuple, list)):
        func_list = func
    else:
        func_list = [func]

    def validator(self, attribute, value):
        if value is not None:
            for func in func_list:
                func(self, attribute, value)

    return validator


@attr.s(auto_attribs=True)
class Epoch:
    start_time: Time = attr.ib(validator=[non_negative, finite])
    end_time: Time = attr.ib(default=None, validator=optional(non_negative))
    initial_size: Size = attr.ib(default=None, validator=optional([positive, finite]))
    final_size: Size = attr.ib(default=None, validator=optional([positive, finite]))

    def __attrs_post_init__(self):
        if self.initial_size is None and self.final_size is None:
            raise ValueError("must set either initial_size or final_size")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            raise ValueError("must have start_time < end_time")
        if self.final_size is None:
            self.final_size = self.initial_size

    @property
    def dt(self):
        return self.end_time - self.start_time


@attr.s(auto_attribs=True)
class Migration:
    source: ID = attr.ib()
    dest: ID = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])
    rate: Rate = attr.ib(validator=[non_negative, finite])

    def __attrs_post_init__(self):
        if self.source == self.dest:
            raise ValueError("source and dest cannot be the same deme")


@attr.s(auto_attribs=True)
class Pulse:
    source: ID = attr.ib()
    dest: ID = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])
    proportion: Proportion = attr.ib(validator=unit_interval)

    def __attrs_post_init__(self):
        if self.source == self.dest:
            raise ValueError("source and dest cannot be the same deme")


@attr.s(auto_attribs=True)
class Deme:
    id: ID = attr.ib()
    ancestor: ID = attr.ib()
    epochs: List[Epoch] = attr.ib()

    @epochs.validator
    def _check_epochs(self, attribute, value):
        if len(self.epochs) != 1:
            raise ValueError(
                "Deme must be created with exactly one epoch."
                "Use add_epoch() to supply additional epochs."
            )

    def __attrs_post_init__(self):
        if self.id == self.ancestor:
            raise ValueError(f"{self.id} cannot be its own ancestor")

    def add_epoch(self, epoch: Epoch):
        assert len(self.epochs) > 0
        prev_epoch = self.epochs[len(self.epochs) - 1]
        if epoch.start_time < prev_epoch.start_time:
            raise ValueError(
                "epochs must be non overlapping and added in time-increasing order"
            )
        if epoch.end_time is None:
            epoch.end_time = prev_epoch.end_time
        prev_epoch.end_time = epoch.start_time
        if epoch.initial_size is None:
            epoch.initial_size = prev_epoch.final_size
        if epoch.final_size is None:
            epoch.final_size = epoch.initial_size
        self.epochs.append(epoch)

    @property
    def start_time(self):
        return self.epochs[0].start_time

    @property
    def end_time(self):
        return self.epochs[-1].end_time

    @property
    def dt(self):
        return self.end_time - self.start_time


@attr.s(auto_attribs=True)
class DemeGraph:
    description: str = attr.ib()
    time_units: str = attr.ib()
    generation_time: Time = attr.ib(validator=[positive, finite])
    default_Ne: Size = attr.ib(default=None, validator=optional([positive, finite]))
    doi: str = attr.ib(default=None)
    demes: List[Deme] = attr.ib(factory=list)
    migrations: List[Migration] = attr.ib(factory=list)
    pulses: List[Pulse] = attr.ib(factory=list)

    def __attrs_post_init__(self):
        self._deme_map: Mapping[ID, Deme] = dict()

    def __getitem__(self, deme_id):
        return self._deme_map[deme_id]

    def __contains__(self, deme_id):
        return deme_id in self._deme_map

    def deme(
        self,
        id,
        ancestor=None,
        start_time=0,
        end_time=float("inf"),
        initial_size=None,
        final_size=None,
        epochs=None,
    ):
        """
        Add deme to the graph.
        """
        if initial_size is None:
            initial_size = self.default_Ne
            if initial_size is None:
                raise ValueError(f"must set initial_size for {id}")
        if final_size is None:
            final_size = initial_size
        if ancestor is not None:
            if ancestor not in self:
                raise ValueError(f"{ancestor} not in deme graph")
            end_time = self[ancestor].epochs[0].start_time
        epoch = Epoch(
            start_time, end_time, initial_size=initial_size, final_size=final_size
        )
        deme = Deme(id, ancestor, [epoch])
        if epochs is not None:
            for epoch in epochs:
                deme.add_epoch(epoch)
        self._deme_map[deme.id] = deme
        self.demes.append(deme)

    def check_time_intersection(self, deme1, deme2, time, closed=False):
        deme1 = self[deme1]
        deme2 = self[deme2]
        time_lo = max(deme1.start_time, deme2.start_time)
        time_hi = min(deme1.end_time, deme2.end_time)
        if time is not None:
            if (not closed and not (time_lo <= time < time_hi)) or (
                closed and not (time_lo <= time <= time_hi)
            ):
                bracket = "]" if closed else ")"
                raise ValueError(
                    f"{time} not in interval [{time_lo}, {time_hi}{bracket}, "
                    f"as defined by the time-intersection of {deme1} and {deme2}."
                )
        return time_lo, time_hi

    def symmetric_migration(self, *demes, rate=0, start_time=None, end_time=None):
        """
        Add continuous (symmetric) migration from ``start_time``.
        """
        if len(demes) < 2:
            raise ValueError("must specify two or more demes")
        for source, dest in itertools.permutations(demes, 2):
            self.migration(source, dest, rate, start_time, end_time)

    def migration(self, source, dest, rate=0, start_time=None, end_time=None):
        """
        Add continuous migration from ``start_time``.
        """
        for deme_id in (source, dest):
            if deme_id not in self:
                raise ValueError(f"{deme_id} not in deme graph")
        time_lo, time_hi = self.check_time_intersection(source, dest, start_time)
        if start_time is None:
            start_time = time_lo
        self.migrations.append(Migration(source, dest, start_time, rate))
        if end_time is not None:
            self.check_time_intersection(source, dest, end_time)
            self.migrations.append(Migration(source, dest, end_time, 0))

    def pulse(self, source, dest, proportion, time):
        """
        Add a pulse of migration at a fixed point in ``time``.
        """
        for deme_id in (source, dest):
            if deme_id not in self:
                raise ValueError(f"{deme_id} not in deme graph")
        if self[source].ancestor == dest or self[dest].ancestor == source:
            raise ValueError(f"{source} and {dest} have ancestor/descendent relation")
        self.check_time_intersection(source, dest, time, closed=True)
        self.pulses.append(Pulse(source, dest, time, proportion))

    def subgraph(
        self,
        deme_id,
        ancestors: List[ID],
        proportions: List[Rate],
        start_time=0,
        end_time=None,
        initial_size=None,
        final_size=None,
        epochs=None,
    ):
        """
        Add a new deme to the graph. The new deme may have multiple ``ancestors``,
        which connect the new deme to the graph via pulse migrations with the
        supplied ``proportions``.
        """
        if len(ancestors) != len(proportions):
            raise ValueError("len(ancestors) != len(proportions)")
        if not math.isclose(sum(proportions), 1.0):
            raise ValueError("proportions must sum to 1")
        self.deme(
            deme_id,
            start_time=start_time,
            end_time=end_time,
            initial_size=initial_size,
            final_size=final_size,
            epochs=epochs,
        )
        for j, ancestor in enumerate(ancestors):
            p = proportions[j] / sum(proportions[j:])
            self.pulse(source=deme_id, dest=ancestor, proportion=p, time=end_time)
        assert p == 1

    def asdict(self):
        """
        Return a dict representation of the deme graph.
        """
        return attr.asdict(self)

    def asdict_compact(self):
        """
        Return a dict representation of the deme graph, with default and
        implicit values removed.
        """
        d = dict(
            description=self.description,
            time_units=self.time_units,
            generation_time=self.generation_time,
        )
        if self.doi is not None:
            d.update(doi=self.doi)
        if self.default_Ne is not None:
            d.update(default_Ne=self.default_Ne)

        assert len(self.demes) > 0
        d.update(demes=dict())
        for deme in self.demes:
            deme_dict = dict()
            if deme.ancestor is not None:
                deme_dict.update(ancestor=deme.ancestor)
            assert len(deme.epochs) > 0
            if deme.epochs[0].start_time > 0:
                deme_dict.update(start_time=deme.epochs[0].start_time)
            deme_dict.update(initial_size=deme.epochs[0].initial_size)
            if deme.epochs[0].final_size != deme.epochs[0].initial_size:
                deme_dict.update(final_size=deme.epochs[0].final_size)

            e_list = []
            for j, epoch in enumerate(deme.epochs):
                e = dict()
                if epoch.start_time > 0:
                    e.update(start_time=epoch.start_time)
                if epoch.initial_size == epoch.final_size:
                    e.update(initial_size=epoch.initial_size)
                else:
                    e.update(final_size=epoch.final_size)
                    if (
                        j == 0
                        or j == len(deme.epochs) - 1
                        or epoch.initial_size != deme.epochs[j - 1].final_size
                    ):
                        e.update(initial_size=epoch.initial_size)
                e_list.append(e)
            if len(e_list) > 1:
                deme_dict.update(epochs=e_list[1:])
            d["demes"][deme.id] = deme_dict

        if len(self.migrations) > 0:
            m_dict = collections.defaultdict(list)
            for migration in self.migrations:
                m_dict[(migration.source, migration.dest)].append(migration)

            m_list = []
            for (source, dest), m_sublist in m_dict.items():
                time_lo, time_hi = self.check_time_intersection(source, dest, None)
                while True:
                    migration, m_sublist = m_sublist[0], m_sublist[1:]
                    m_list.append(dict(source=source, dest=dest, rate=migration.rate))
                    if migration.time != time_lo:
                        m_list[-1].update(start_time=migration.time)
                    if len(m_sublist) == 0:
                        break
                    if migration.rate != 0 and m_sublist[0].rate == 0:
                        if m_sublist[0].time != time_hi:
                            m_list[-1].update(end_time=m_sublist[0].time)
                        m_sublist = m_sublist[1:]
                        if len(m_sublist) == 0:
                            break
            # TODO collapse into symmetric migrations
            d.update(migrations=m_list)

        if len(self.pulses) > 0:
            d.update(pulses=[attr.asdict(pulse) for pulse in self.pulses])

        return d
