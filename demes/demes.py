import typing
from typing import List, Mapping, Union, Optional
import itertools
import math
import collections
import copy
import operator

import attr
from attr.validators import optional

from .script import dumps, loads

Number = Union[int, float]
ID = str
Time = Number
Size = Number
Rate = float
Proportion = float

_ISCLOSE_REL_TOL = 1e-9
_ISCLOSE_ABS_TOL = 1e-12

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


def isclose(
    a: Optional[Number],
    b: Optional[Number],
    *,
    rel_tol=_ISCLOSE_REL_TOL,
    abs_tol=_ISCLOSE_ABS_TOL,
) -> bool:
    """
    Wrapper around math.isclose() that handles None.
    """
    if None in (a, b):
        return (a,) == (b,)
    else:
        return math.isclose(
            typing.cast(float, a),
            typing.cast(float, b),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )


def isclose_deme_proportions(
    a_ids: Optional[List[ID]],
    a_proportions: Optional[List[Proportion]],
    b_ids: Optional[List[ID]],
    b_proportions: Optional[List[Proportion]],
    *,
    rel_tol=_ISCLOSE_REL_TOL,
    abs_tol=_ISCLOSE_ABS_TOL,
) -> bool:
    """
    Returns true if (a_ids, a_proportions) and (b_ids, b_proportions)
    are semantically equivalent. The order of ids is ignored, and proportions
    are checked for numerical closeness.
    """
    if None in (a_ids, b_ids):
        return (a_ids, a_proportions) == (b_ids, b_proportions)

    a_ids = typing.cast(List[ID], a_ids)
    a_proportions = typing.cast(List[Proportion], a_proportions)
    b_ids = typing.cast(List[ID], b_ids)
    b_proportions = typing.cast(List[Number], b_proportions)

    if len(a_ids) != len(b_ids) or len(a_proportions) != len(b_proportions):
        return False
    a = sorted(zip(a_ids, a_proportions), key=operator.itemgetter(0))
    b = sorted(zip(b_ids, b_proportions), key=operator.itemgetter(0))
    for (a_id, a_proportion), (b_id, b_proportion) in zip(a, b):
        if a_id != b_id or not isclose(
            a_proportion, b_proportion, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return False
    return True


@attr.s(auto_attribs=True)
class Epoch:
    """
    Population size parameters for a deme in a specified time period.
    Times follow the forwards-in-time convention (time values increase
    from the present towards the past). The start time of the epoch is
    the more ancient time, and the end time is more recent, so that the
    start time must be greater than the end time

    :ivar start_time: The start time of the epoch.
    :ivar end_time: The end time of the epoch (must be specified).
    :ivar initial_size: Population size at ``start_time``.
    :ivar final_size: Population size at ``end_time``.
        If ``initial_size != final_size``, the population size changes
        monotonically between the start and end times.
    :ivar size_function: The size change function. Common options are constant,
        exponential, or linear, though any string is valid. Warning: downstream
        simulators might not understand the size_function provided.
    :ivar selfing_rate: An optional selfing rate for this epoch.
    :ivar cloning_rate: An optional cloning rate for this epoch.
    """

    start_time: Optional[Time] = attr.ib(default=None, validator=optional(non_negative))
    end_time: Time = attr.ib(default=None, validator=[non_negative, finite])
    initial_size: Optional[Size] = attr.ib(
        default=None, validator=optional([positive, finite])
    )
    final_size: Optional[Size] = attr.ib(
        default=None, validator=optional([positive, finite])
    )
    size_function: Optional[str] = attr.ib(default=None)
    selfing_rate: Optional[Proportion] = attr.ib(
        default=None, validator=optional(unit_interval)
    )
    cloning_rate: Optional[Proportion] = attr.ib(
        default=None, validator=optional(unit_interval)
    )

    def __attrs_post_init__(self):
        if self.initial_size is None and self.final_size is None:
            raise ValueError("must set either initial_size or final_size")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time <= self.end_time
        ):
            raise ValueError("must have start_time > end_time")
        if (
            self.start_time is not None
            and self.initial_size is not None
            and self.final_size is not None
        ):
            if math.isinf(self.start_time) and self.initial_size != self.final_size:
                raise ValueError("if start time is inf, must be a constant size epoch")

    @property
    def time_span(self):
        """
        The time span of the epoch.
        """
        return self.start_time - self.end_time

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and isclose(
                self.start_time, other.start_time, rel_tol=rel_tol, abs_tol=abs_tol
            )
            and isclose(self.end_time, other.end_time, rel_tol=rel_tol, abs_tol=abs_tol)
            and isclose(
                self.initial_size, other.initial_size, rel_tol=rel_tol, abs_tol=abs_tol
            )
            and isclose(
                self.final_size, other.final_size, rel_tol=rel_tol, abs_tol=abs_tol
            )
            and self.size_function == other.size_function
            and isclose(
                self.selfing_rate, other.selfing_rate, rel_tol=rel_tol, abs_tol=abs_tol
            )
            and isclose(
                self.cloning_rate, other.cloning_rate, rel_tol=rel_tol, abs_tol=abs_tol
            )
        )


@attr.s(auto_attribs=True)
class Migration:
    """
    Parameters for continuous migration from one deme to another.
    Source and destination demes follow the forwards-in-time convention,
    of migrations born in the source deme having children in the dest
    deme.

    :ivar source: The source deme.
    :ivar dest: The destination deme.
    :ivar start_time: The time at which the migration rate becomes activate.
    :ivar end_time: The time at which the migration rate is deactivated.
    :ivar rate: The rate of migration. Set to zero to disable migrations after
        the given time.
    """

    source: ID = attr.ib()
    dest: ID = attr.ib()
    start_time: Time = attr.ib(validator=non_negative)
    end_time: Time = attr.ib(validator=[non_negative, finite])
    rate: Rate = attr.ib(validator=[non_negative, finite])

    def __attrs_post_init__(self):
        if self.source == self.dest:
            raise ValueError("source and dest cannot be the same deme")

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and self.source == other.source
            and self.dest == other.dest
            and isclose(
                self.start_time, other.start_time, rel_tol=rel_tol, abs_tol=abs_tol
            )
            and isclose(self.end_time, other.end_time, rel_tol=rel_tol, abs_tol=abs_tol)
            and isclose(self.rate, other.rate, rel_tol=rel_tol, abs_tol=abs_tol)
        )


@attr.s(auto_attribs=True)
class Pulse:
    """
    Parameters for a pulse of migration from one deme to another.
    Source and destination demes follow the forwards-in-time convention,
    of migrations born in the source deme having children in the dest
    deme.

    :ivar source: The source deme.
    :ivar dest: The destination deme.
    :ivar time: The time of migration.
    :ivar proportion: At the instant after migration, this is the proportion
        of individuals in the destination deme made up of individuals from
        the source deme.
    """

    source: ID = attr.ib()
    dest: ID = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])
    proportion: Proportion = attr.ib(validator=unit_interval)

    def __attrs_post_init__(self):
        if self.source == self.dest:
            raise ValueError("source and dest cannot be the same deme")

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and self.source == other.source
            and self.dest == other.dest
            and isclose(self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol)
            and isclose(
                self.proportion, other.proportion, rel_tol=rel_tol, abs_tol=abs_tol
            )
        )


@attr.s(auto_attribs=True)
class Split:
    """
    Parameters for a split event, in which a deme ends at a given time and
    contributes ancestry to an arbitrary number of descendant demes. Note
    that there could be just a single descendant deme, in which case ``split``
    is a bit of a misnomer...

    :ivar parent: The parental deme.
    :ivar children: A list of descendant demes.
    :ivar time: The split time.
    """

    parent: ID = attr.ib()
    children: List[ID] = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])

    def __attrs_post_init__(self):
        if not isinstance(self.children, list):
            raise ValueError("children of split must be passed as a list")
        for child in self.children:
            if child == self.parent:
                raise ValueError("child and parent cannot be the same deme")

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and self.parent == other.parent
            and sorted(self.children) == sorted(other.children)
            and isclose(self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol)
        )


@attr.s(auto_attribs=True)
class Branch:
    """
    Parameters for a branch event, where a new deme branches off from a parental
    deme. The parental deme need not end at that time.

    :ivar parent: The parental deme.
    :ivar child: The descendant deme.
    :ivar time: The branch time.
    """

    parent: ID = attr.ib()
    child: ID = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])

    def __attrs_post_init__(self):
        if self.child == self.parent:
            raise ValueError("child and parent cannot be the same deme")

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and self.parent == other.parent
            and self.child == other.child
            and isclose(self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol)
        )


@attr.s(auto_attribs=True)
class Merge:
    """
    Parameters for a merge event, in which two or more demes end at some time and
    contribute to a descendant deme.

    :ivar parents: A list of parental demes.
    :ivar proportions: A list of ancestry proportions, in order of `parents`.
    :ivar child: The descendant deme.
    :ivar time: The merge time.
    """

    parents: List[ID] = attr.ib()
    proportions: List[Proportion] = attr.ib()
    child: ID = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])

    def __attrs_post_init__(self):
        if not isinstance(self.parents, list):
            raise ValueError("parents must be passed as a list")
        if not isinstance(self.proportions, list):
            raise ValueError("proportions must be passed as a list")
        if len(self.parents) < 2:
            raise ValueError("merge must involve at least two ancestors")
        if math.isclose(sum(self.proportions), 1) is False:
            raise ValueError("proportions must sum to 1")
        if len(self.parents) != len(self.proportions):
            raise ValueError("parents and proportions must have same length")
        if self.child in self.parents:
            raise ValueError("merged deme cannot be its own ancestor")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("cannot repeat parents in merge")

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and isclose_deme_proportions(
                self.parents,
                self.proportions,
                other.parents,
                other.proportions,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            and self.child == other.child
            and isclose(self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol)
        )


@attr.s(auto_attribs=True)
class Admix:
    """
    Parameters for an admixture event, where two or more demes contribute ancestry
    to a new deme.

    :ivar parents: A list of source demes.
    :ivar proportions: A list of ancestry proportions, in order of `parents`.
    :ivar child: The admixed deme.
    :ivar time: The admixture time.
    """

    parents: List[ID] = attr.ib()
    proportions: List[Proportion] = attr.ib()
    child: ID = attr.ib()
    time: Time = attr.ib(validator=[non_negative, finite])

    def __attrs_post_init__(self):
        if not isinstance(self.parents, list):
            raise ValueError("parents must be passed as a list")
        if not isinstance(self.proportions, list):
            raise ValueError("proportions must be passed as a list")
        if len(self.parents) < 2:
            raise ValueError("admixture must involve at least two ancestors")
        if math.isclose(sum(self.proportions), 1) is False:
            raise ValueError("Proportions must sum to 1")
        if len(self.parents) != len(self.proportions):
            raise ValueError("parents and proportions must have same length")
        if self.child in self.parents:
            raise ValueError("admixed deme cannot be its own ancestor")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("cannot repeat parents in admixure")

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and isclose_deme_proportions(
                self.parents,
                self.proportions,
                other.parents,
                other.proportions,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            and self.child == other.child
            and isclose(self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol)
        )


@attr.s(auto_attribs=True)
class Deme:
    """
    A collection of individuals that are exchangeable at any fixed time.

    :ivar id: A string identifier for the deme.
    :ivar description: An optional description of the deme.
    :ivar ancestors: List of ancestors to the deme.
        If the deme has no ancestors, this should be ``None``, and cannot
        be used with ``ancestor``.
    :ivar proportions: List of proportions if ``ancestors`` is given.
    :ivar epochs: A list of epochs, which define the population size(s) of
        the deme. The deme must be initially created with exactly one epoch.
        Additional epochs may be added with :meth:`.add_epoch`
    :vartype epochs: list of :class:`.Epoch`
    :ivar selfing_rate: An optional selfing rate for this deme.
    :ivar cloning_rate: An optional cloning rate for this deme.
    """

    id: ID = attr.ib()
    description: str = attr.ib()
    ancestors: List[ID] = attr.ib()
    proportions: List[Proportion] = attr.ib()
    epochs: List[Epoch] = attr.ib()
    selfing_rate: Optional[Proportion] = attr.ib(
        default=None, validator=optional([unit_interval])
    )
    cloning_rate: Optional[Proportion] = attr.ib(
        default=None, validator=optional([unit_interval])
    )

    @epochs.validator
    def _check_epochs(self, attribute, value):
        if len(self.epochs) != 1:
            raise ValueError(
                "Deme must be created with exactly one epoch."
                "Use add_epoch() to supply additional epochs."
            )

    def __attrs_post_init__(self):
        if self.ancestors is not None:
            if not isinstance(self.ancestors, (list, tuple)):
                raise TypeError("ancestors must be a list of deme IDs")
            if len(self.ancestors) > 1 and self.proportions is None:
                raise ValueError("proportions must be set if more than one ancestor")
            if len(self.ancestors) != len(self.proportions):
                raise ValueError("ancestors and proportions must have same length")
            if self.id in self.ancestors:
                raise ValueError(f"{self.id} cannot be its own ancestor")
        # if selfing or cloning rates are not given, set them to deme's default rate
        epoch = self.epochs[0]
        if epoch.selfing_rate is None:
            epoch.selfing_rate = self.selfing_rate
        if epoch.cloning_rate is None:
            epoch.cloning_rate = self.cloning_rate

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        return (
            self.__class__ is other.__class__
            and self.id == other.id
            and isclose_deme_proportions(
                self.ancestors,
                self.proportions,
                other.ancestors,
                other.proportions,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            and all(
                e1.isclose(e2, rel_tol=rel_tol, abs_tol=abs_tol)
                for e1, e2 in zip(self.epochs, other.epochs)
            )
        )

    def add_epoch(self, epoch: Epoch):
        """
        Add an epoch to the deme's epoch list.
        Epochs must be non overlapping and added in time-decreasing order, i.e.
        starting with the most ancient epoch and adding epochs sequentially toward
        present.

        :param .Epoch epoch: The epoch to add.
        """
        assert len(self.epochs) > 0
        # if the epoch start time is not given, it equals the previous epoch's end time
        prev_epoch = self.epochs[-1]
        if epoch.start_time is None:
            epoch.start_time = prev_epoch.end_time
        elif epoch.start_time > prev_epoch.end_time:
            raise ValueError(
                "epochs must be non overlapping and added in time-decreasing order"
            )
        if prev_epoch.end_time != epoch.start_time:
            raise ValueError("cannot have gap between consecutive epochs")
        if epoch.time_span <= 0:
            raise ValueError("epoch must exist for some positive time")
        # implicitly set the initial and final sizes, if not given
        if epoch.initial_size is None:
            epoch.initial_size = prev_epoch.final_size
        if epoch.final_size is None:
            epoch.final_size = epoch.initial_size
        # check or assign the size function over this epoch
        if epoch.size_function is None:
            if epoch.initial_size == epoch.final_size:
                epoch.size_function = "constant"
            else:
                epoch.size_function = "exponential"
        else:
            # check if constant function is correct
            if (
                epoch.size_function == "constant"
                and epoch.initial_size != epoch.final_size
            ):
                raise ValueError(
                    "epoch size function is constant but initial and "
                    "final sizes are not equal"
                )
        # if selfing or cloning rates are not given, set them to deme's default rate
        if epoch.selfing_rate is None:
            epoch.selfing_rate = self.selfing_rate
        if epoch.cloning_rate is None:
            epoch.cloning_rate = self.cloning_rate
        self.epochs.append(epoch)

    @property
    def start_time(self):
        """
        The start time of the deme's existence.
        """
        return self.epochs[0].start_time

    @property
    def end_time(self):
        """
        The end time of the deme's existence.
        """
        return self.epochs[-1].end_time

    @property
    def time_span(self):
        """
        The time span over which the deme exists.
        """
        return self.start_time - self.end_time


@attr.s(auto_attribs=True)
class DemeGraph:
    """
    A directed graph that describes a demography. Vertices are demes and edges
    correspond to ancestor/descendent relations. Edges are directed from
    ancestors to descendants.

    :ivar description: A human readable description of the demography.
    :ivar time_units: The units of time used for the demography. This is
        commonly ``years`` or ``generations``, but can be any string.
        This field is intended to be useful for documenting a demography,
        but the actual value provided here should not be relied upon.
    :ivar generation_time: The generation time of demes, in units given
        by the ``time_units`` parameter. Concretely, dividing all times
        by ``generation_time`` will convert the deme graph to have time
        units in generations.  If ``generation_time`` is ``None``, the units
        are assumed to be in generations already.
        See also: :meth:`.in_generations`.
    :ivar default_Ne: The default population size to use when creating new
        demes with :meth:`.deme`. May be ``None``.
    :ivar doi: If the deme graph describes a published demography, the DOI
        should be be given here. May be ``None``.
    :ivar demes: A list of demes in the demography.
        Not intended to be passed when the deme graph is instantiated.
        Use :meth:`.deme` instead.
    :vartype demes: list of :class:`.Deme`
    :ivar migrations: A list of continuous migrations for the demography.
        Not intended to be passed when the deme graph is instantiated.
        Use :meth:`migration` or :meth:`symmetric_migration` instead.
    :vartype migrations: list of :class:`.Migration`
    :ivar pulses: A list of migration pulses for the demography.
        Not intended to be passed when the deme graph is instantiated.
        Use :meth:`pulse` instead.
    """

    description: str = attr.ib()
    time_units: str = attr.ib()
    generation_time: Optional[Time] = attr.ib(
        default=None, validator=optional([positive, finite])
    )
    default_Ne: Optional[Size] = attr.ib(
        default=None, validator=optional([positive, finite])
    )
    doi: Optional[str] = attr.ib(default=None)
    demes: List[Deme] = attr.ib(factory=list)
    migrations: List[Migration] = attr.ib(factory=list)
    pulses: List[Pulse] = attr.ib(factory=list)
    splits: List[Split] = attr.ib(factory=list)
    branches: List[Branch] = attr.ib(factory=list)
    mergers: List[Merge] = attr.ib(factory=list)
    admixtures: List[Admix] = attr.ib(factory=list)
    selfing_rate: Proportion = attr.ib(default=None)
    cloning_rate: Proportion = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._deme_map: Mapping[ID, Deme] = dict()

    def __getitem__(self, deme_id):
        """
        Return the :class:`.Deme` with the specified id.
        """
        return self._deme_map[deme_id]

    def __contains__(self, deme_id):
        """
        Check if the deme graph contains a deme with the specified id.
        """
        return deme_id in self._deme_map

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        def sorted_eq(aa, bb, *, rel_tol, abs_tol) -> bool:
            # Order-agnostic equality check.
            if len(aa) != len(bb):
                return False
            for (a, b) in zip(sorted(aa), sorted(bb)):
                if not a.isclose(b, rel_tol=rel_tol, abs_tol=abs_tol):
                    return False
            return True

        return (
            self.__class__ is other.__class__
            and self.time_units == other.time_units
            and self.generation_time == other.generation_time
            and sorted_eq(self.demes, other.demes, rel_tol=rel_tol, abs_tol=abs_tol)
            and sorted_eq(
                self.migrations, other.migrations, rel_tol=rel_tol, abs_tol=abs_tol
            )
            and sorted_eq(self.pulses, other.pulses, rel_tol=rel_tol, abs_tol=abs_tol)
        )

    def deme(
        self,
        id,
        description=None,
        ancestors=None,
        proportions=None,
        start_time=None,
        end_time=None,
        initial_size=None,
        final_size=None,
        epochs=None,
        selfing_rate=None,
        cloning_rate=None,
    ):
        """
        Add a deme to the graph.

        :param str id: A string identifier for the deme.
        :param list ancestors: A list of ancestors of this deme. May be ``None``.
            If ``len(ancestors) > 1``, must also give ``proportions``.
        :param list proportions: A list of proportions of ancestory for ``ancestors``.
            Proportions must sum to 1.
        :param start_time: The time at which this deme begins existing.
        :param end_time: The time at which this deme stops existing.
            If the deme has an ancestor the ``end_time`` will be set to the
            ancestor's ``start_time``.
        :param initial_size: The initial population size of the deme. If ``None``,
            this is taken from the deme graph's ``default_Ne`` field.
        :param final_size: The final population size of the deme. If ``None``,
            the deme has a constant ``initial_size`` population size.
        :param epochs: Additional epochs that define population size changes for
            the deme.
        :ivar selfing_rate: An optional selfing rate for this deme.
        :ivar cloning_rate: An optional cloning rate for this deme.
        """
        if initial_size is None:
            initial_size = self.default_Ne
        if initial_size is None and epochs is not None:
            initial_size = epochs[0].initial_size
        if initial_size is None:
            raise ValueError(f"must set initial_size for {id}")
        if selfing_rate is None:
            selfing_rate = self.selfing_rate
        if cloning_rate is None:
            cloning_rate = self.cloning_rate
        # set the start time to inf or to the ancestors end times, if not given
        if ancestors is not None:
            if not isinstance(ancestors, (list, tuple)):
                raise TypeError("ancestors must be a list of deme IDs")
            if len(ancestors) > 1 and start_time is None:
                raise ValueError("must specify start time if more than one ancestor")
            if ancestors[0] in self and start_time is None:
                start_time = self[ancestors[0]].epochs[-1].end_time
            if len(ancestors) == 1 and proportions is None:
                proportions = [1.0]
        else:
            if start_time is None:
                start_time = float("inf")
        # build the deme, and then add epochs as necessary
        if epochs is None:
            # if epochs are not given, we assign a single epoch over that deme
            if final_size is None:
                final_size = initial_size
            if end_time is None:
                end_time = 0
            if initial_size == final_size:
                size_function = "constant"
            else:
                size_function = "exponential"
            epoch = Epoch(
                start_time,
                end_time,
                initial_size=initial_size,
                final_size=final_size,
                size_function=size_function,
                selfing_rate=selfing_rate,
                cloning_rate=cloning_rate,
            )
            deme = Deme(
                id,
                description,
                ancestors,
                proportions,
                [epoch],
                selfing_rate,
                cloning_rate,
            )
        else:
            if end_time is None:
                end_time = epochs[-1].end_time
            if end_time != epochs[-1].end_time:
                raise ValueError("deme and final epoch end times do not align")
            if epochs[0].selfing_rate is None:
                epochs[0].selfing_rate = selfing_rate
            if epochs[0].cloning_rate is None:
                epochs[0].cloning_rate = cloning_rate
            # deal with first epoch and deme start times
            if epochs[0].start_time is None:
                # first epoch starts at deme start time
                epochs[0].start_time = start_time
            elif epochs[0].start_time < start_time:
                # insert const size epoch to reach the start of first listed epoch
                epochs.insert(
                    0,
                    Epoch(
                        start_time,
                        epochs[0].start_time,
                        initial_size=initial_size,
                        final_size=initial_size,
                        size_function="constant",
                        selfing_rate=selfing_rate,
                        cloning_rate=cloning_rate,
                    ),
                )
            elif epochs[0].start_time > start_time:
                raise ValueError(
                    "first epoch start time must be less than or equal to "
                    "deme start time"
                )
            # set up sizes of first deme, since subsequent demes are built off of it
            if epochs[0].final_size is None:
                epochs[0].final_size = epochs[0].initial_size
            if epochs[0].size_function is None:
                if epochs[0].initial_size == epochs[0].final_size:
                    epochs[0].size_function = "constant"
                else:
                    epochs[0].size_function = "exponential"
            deme = Deme(
                id,
                description,
                ancestors,
                proportions,
                [epochs[0]],
                selfing_rate,
                cloning_rate,
            )
            for epoch in epochs[1:]:
                deme.add_epoch(epoch)
        self._deme_map[deme.id] = deme
        self.demes.append(deme)

    def check_time_intersection(self, deme1, deme2, time):
        deme1 = self[deme1]
        deme2 = self[deme2]
        time_lo = max(deme1.end_time, deme2.end_time)
        time_hi = min(deme1.start_time, deme2.start_time)
        if time is not None:
            if not (time_lo <= time <= time_hi):
                raise ValueError(
                    f"{time} not in interval [{time_lo}, {time_hi}], "
                    f"as defined by the time-intersection of {deme1.id} "
                    f"(start_time={deme1.start_time}, end_time={deme1.end_time}) "
                    f"and {deme2.id} (start_time={deme2.start_time}, "
                    f"end_time={deme2.end_time})."
                )
        return time_lo, time_hi

    def symmetric_migration(self, demes=[], rate=0, start_time=None, end_time=None):
        """
        Add continuous symmetric migrations between all pairs of demes in a list.

        :param demes: list of deme IDs. Migration is symmetric between all
            pairs of demes in this list.
        :param rate: The rate of migration per generation.
        :param start_time: The time at which the migration rate is enabled.
        :param end_time: The time at which the migration rate is disabled.
        """
        if len(demes) < 2:
            raise ValueError("must specify two or more demes")
        for source, dest in itertools.permutations(demes, 2):
            self.migration(source, dest, rate, start_time, end_time)

    def migration(self, source, dest, rate=0, start_time=None, end_time=None):
        """
        Add continuous migration from one deme to another.
        Source and destination demes follow the forwards-in-time convention,
        so that the migration rate refers to the movement of individuals from
        the ``source`` deme to the ``dest`` deme.

        :param source: The source deme.
        :param dest: The destination deme.
        :param rate: The rate of migration per generation.
        :param start_time: The time at which the migration rate is enabled.
            If ``None``, the start time is defined by the earliest time at
            which the demes coexist.
        :param end_time: The time at which the migration rate is disabled.
            If ``None``, the end time is defined by the latest time at which
            the demes coexist.
        """
        for deme_id in (source, dest):
            if deme_id not in self:
                raise ValueError(f"{deme_id} not in deme graph")
        time_lo, time_hi = self.check_time_intersection(source, dest, start_time)
        if start_time is None:
            start_time = time_hi
        else:
            self.check_time_intersection(source, dest, start_time)
        if end_time is None:
            end_time = time_lo
        else:
            self.check_time_intersection(source, dest, end_time)
        self.migrations.append(Migration(source, dest, start_time, end_time, rate))

    def pulse(self, source, dest, proportion, time):
        """
        Add a pulse of migration at a fixed time.
        Source and destination demes follow the forwards-in-time convention.

        :param source: The source deme.
        :param dest: The destination deme.
        :param proportion: At the instant after migration, this is the expected
            proportion of individuals in the destination deme made up of individuals
            from the source deme.
        :param time: The time at which migrations occur.
        """
        for deme_id in (source, dest):
            if deme_id not in self:
                raise ValueError(f"{deme_id} not in deme graph")
        self.check_time_intersection(source, dest, time)
        self.pulses.append(Pulse(source, dest, time, proportion))

    @property
    def successors(self):
        """
        Lists of successors for all demes in the graph.
        """
        # use collections.defaultdict(list)
        succ = {}
        for deme_info in self.demes:
            succ.setdefault(deme_info.id, [])
            if deme_info.ancestors is not None:
                for a in deme_info.ancestors:
                    succ.setdefault(a, [])
                    succ[a].append(deme_info.id)
        return succ

    @property
    def predecessors(self):
        """
        Lists of predecessors (ancestors) for all demes in the graph.
        """
        pred = {}
        for deme_info in self.demes:
            pred.setdefault(deme_info.id, [])
            if deme_info.ancestors is not None:
                for a in deme_info.ancestors:
                    pred[deme_info.id].append(a)
        return pred

    def split(self, parent, children, time):
        """
        Add split event at a given time. Split events involve a parental deme
        whose end time equals the start time of all children demes.

        :param parent: The ancestral deme.
        :param children: A list of descendant demes.
        :param time: The time at which split occurs.
        """
        for child in children:
            # check parent/children relationship and end/start times
            if child == parent:
                raise ValueError("cannot be ancestor of own deme")
            if self[parent].end_time != self[child].start_time:
                raise ValueError(
                    f"{parent} and {child} must have matching end and start times"
                )
            # the ancestor of each child population is set
            self[child].ancestors = [parent]
        self.splits.append(Split(parent, children, time))

    def branch(self, parent, child, time):
        """
        Add branch event at a given time.

        :param parent: The ancestral deme.
        :param children: A list of descendant demes.
        :param time: The time at which branch event occurs.
        """
        if (
            self[child].start_time < self[parent].end_time
            or self[child].start_time >= self[parent].start_time
        ):
            raise ValueError(
                f"{child} start time must be within {parent} time interval"
            )
        # set the ancestor of the child population
        self[child].ancestors = [parent]
        self.branches.append(Branch(parent, child, time))

    def merge(self, parents, proportions, child, time):
        """
        Add merger event at a given time, where multiple parents contribute to
        a descendant deme, and the parent demes cease to exist at that time.

        :param parents: The ancestral demes.
        :param proportions: Proportions of ancestral demes contributing to descendant.
        :param children: The descendant deme.
        :param time: The time at which merger occurs.
        """
        if self[child].start_time != time:
            raise ValueError(
                f"{child}'s start time must equal admixture time of {time}"
            )
        # for parental populations, we check that their end time is <= merge time
        for parent in parents:
            if self[parent].end_time > time:
                raise ValueError(f"deme {parent} has end time earlier than {time}")
        # if any parent end times are more recent than merge time, we adjust the end
        # and remove epochs that extend beyond that merger time
        for parent in parents:
            if self[parent].end_time < time:
                while self[parent].epochs[-1].end_time < time:
                    if self[parent].epochs[-1].start_time <= time:
                        del self[parent].epochs[-1]
                    else:
                        self[parent].epochs[-1].end_time = time
        # set the ancestors and proportions of the child deme
        self[child].ancestors = parents
        self[child].proportions = proportions
        self.mergers.append(Merge(parents, proportions, child, time))

    def admix(self, parents, proportions, child, time):
        """
        Add admixture event at a given time, where multiple parents contribute to
        a descendant deme, and the parent demes continue to exist beyond that time.

        :param parents: The ancestral demes.
        :param proportions: Proportions of ancestral demes contributing to descendant.
        :param children: The descendant deme.
        :param time: The time at which admixture occurs.
        """
        if self[child].start_time != time:
            raise ValueError(
                f"{child}'s start time must equal admixture time of {time}"
            )
        # for parental populations, we check that their end time is <= admixture time
        for parent in parents:
            if self[parent].end_time > time:
                raise ValueError(f"deme {parent} has end time earlier than {time}")
        # set the ancestors and proportions of the child deme
        self[child].ancestors = parents
        self[child].proportions = proportions
        self.admixtures.append(Admix(parents, proportions, child, time))

    def get_demographic_events(self):
        """
        Loop through successors/predecessors to add splits, branches, mergers,
        and admixtures to the deme graph. If a deme has more than one predecessor,
        then it is a merger or an admixture event, which we differentiate by end and
        start times of those demes. If a deme has a single predecessor, we check
        whether it is a branch (start time != predecessor's end time), or split.

        This is only used when we build a demography from a YAML file, since it
        uses the successors/predecessors that are determined by ancestor relationships.
        """
        splits_to_add = {}
        for c, p in self.predecessors.items():
            if len(p) == 0:
                continue
            elif len(p) == 1:
                if self[c].start_time == self[p[0]].end_time:
                    splits_to_add.setdefault(p[0], set())
                    splits_to_add[p[0]].add(c)
                else:
                    self.branch(p[0], c, self[c].start_time)
            else:
                time_aligned = True
                for deme_from in p:
                    if self[c].start_time != self[deme_from].end_time:
                        time_aligned = False
                if time_aligned is True:
                    self.merge(
                        self[c].ancestors, self[c].proportions, c, self[c].start_time
                    )
                else:
                    self.admix(
                        self[c].ancestors, self[c].proportions, c, self[c].start_time
                    )
        for deme_from, demes_to in splits_to_add.items():
            self.split(deme_from, list(demes_to), self[deme_from].end_time)

    def validate(self):
        """
        Validates the demographic model.
        """
        loads(dumps(self))

    def in_generations(self):
        """
        Return a copy of the demes graph with times in units of generations.
        """
        deme_graph = copy.deepcopy(self)
        deme_graph.time_units = "generations"
        generation_time = self.generation_time
        if generation_time is not None:
            deme_graph.generation_time = None
            for deme in deme_graph.demes:
                for epoch in deme.epochs:
                    epoch.start_time /= generation_time
                    epoch.end_time /= generation_time
            for migration in deme_graph.migrations:
                migration.start_time /= generation_time
                migration.end_time /= generation_time
            for pulse in deme_graph.pulses:
                pulse.time /= generation_time
            deme_graph.splits = []
            deme_graph.branches = []
            deme_graph.mergers = []
            deme_graph.admixtures = []
            deme_graph.get_demographic_events()
        return deme_graph

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
        )
        if self.generation_time is not None:
            d.update(generation_time=self.generation_time)
        if self.doi is not None:
            d.update(doi=self.doi)
        if self.default_Ne is not None:
            d.update(default_Ne=self.default_Ne)

        if self.selfing_rate is not None:
            d.update(selfing_rate=self.selfing_rate)
        if self.cloning_rate is not None:
            d.update(cloning_rate=self.cloning_rate)

        assert len(self.demes) > 0
        d.update(demes=dict())
        for deme in self.demes:
            deme_dict = dict()
            # add ancestors to deme if not None
            if deme.ancestors is not None:
                deme_dict.update(ancestors=deme.ancestors)
                if len(deme.ancestors) > 1:
                    deme_dict.update(proportions=deme.proportions)
                if any([deme.start_time != self[a].end_time for a in deme.ancestors]):
                    deme_dict.update(start_time=deme.start_time)
            else:
                # corner case of no ancestors but finite start time
                if math.isfinite(deme.start_time):
                    deme_dict.update(start_time=deme.start_time)
            # add selfing and cloning rates, if not None
            if deme.selfing_rate is not None:
                if self.selfing_rate is None or (
                    self.selfing_rate is not None
                    and deme.selfing_rate != self.selfing_rate
                ):
                    deme_dict.update(selfing_rate=deme.selfing_rate)
            if deme.cloning_rate is not None:
                if self.cloning_rate is None or (
                    self.cloning_rate is not None
                    and deme.cloning_rate != self.cloning_rate
                ):
                    deme_dict.update(cloning_rate=deme.cloning_rate)

            assert len(deme.epochs) > 0
            e_list = []
            for j, epoch in enumerate(deme.epochs):
                e = dict()
                # end time required for epochs
                e.update(end_time=epoch.end_time)
                e.update(initial_size=epoch.initial_size)
                if epoch.final_size != epoch.initial_size:
                    e.update(final_size=epoch.final_size)
                if epoch.size_function not in ["constant", "exponential"]:
                    e.update(size_function=epoch.size_function)
                if epoch.selfing_rate is not None:
                    if deme.selfing_rate is not None:
                        if epoch.selfing_rate != deme.selfing_rate:
                            e.update(selfing_rate=epoch.selfing_rate)
                    elif self.selfing_rate is not None:
                        if epoch.selfing_rate != self.selfing_rate:
                            e.update(selfing_rate=epoch.selfing_rate)
                    else:
                        e.update(selfing_rate=epoch.selfing_rate)
                if epoch.cloning_rate is not None:
                    if deme.cloning_rate is not None:
                        if epoch.cloning_rate != deme.cloning_rate:
                            e.update(cloning_rate=epoch.cloning_rate)
                    elif self.cloning_rate is not None:
                        if epoch.cloning_rate != self.cloning_rate:
                            e.update(cloning_rate=epoch.cloning_rate)
                    else:
                        e.update(cloning_rate=epoch.cloning_rate)
                e_list.append(e)
            if len(e_list) > 1:
                # if more than one epoch, list all epochs
                deme_dict.update(epochs=e_list)
            else:
                # if a single epoch, don't list as under epochs
                deme_dict.update(initial_size=e_list[0]["initial_size"])
                if "final_size" in e_list[0]:
                    if e_list[0]["final_size"] != e_list[0]["initial_size"]:
                        deme_dict.update(final_size=e_list[0]["final_size"])
                if e_list[0]["end_time"] > 0:
                    deme_dict.update(end_time=e_list[0]["end_time"])
            if deme.description is not None:
                deme_dict.update(description=deme.description)
            d["demes"][deme.id] = deme_dict

        if len(self.migrations) > 0:
            m_dict = collections.defaultdict(list)
            for migration in self.migrations:
                m_dict[(migration.source, migration.dest)].append(
                    dict(rate=migration.rate)
                )
                time_lo, time_hi = self.check_time_intersection(
                    migration.source, migration.dest, None
                )
                if migration.end_time != time_lo:
                    m_dict[(migration.source, migration.dest)][-1].update(
                        end_time=migration.end_time
                    )
                if migration.start_time != time_hi:
                    m_dict[(migration.source, migration.dest)][-1].update(
                        start_time=migration.start_time
                    )
            # collapse into symmetric and asymmetric migrations
            m_symmetric = []
            m_asymmetric = []
            for (source, dest), m_list in m_dict.items():
                # check if there is equal, reverse migration over the same epoch
                if (dest, source) in m_dict:
                    for m in m_list:
                        no_symmetry = True
                        for i, m_compare in enumerate(m_dict[(dest, source)]):
                            if m == m_compare:
                                m_symmetric.append(
                                    dict(demes=[source, dest], rate=m["rate"])
                                )
                                if "start_time" in m:
                                    m_symmetric[-1]["start_time"] = m["start_time"]
                                if "end_time" in m:
                                    m_symmetric[-1]["end_time"] = m["end_time"]
                                # pop the m_compare so we don't repeat it
                                m_dict[(dest, source)].remove(m_compare)
                                no_symmetry = False
                                break
                        if no_symmetry:
                            m_asymmetric.append(
                                dict(source=source, dest=dest, rate=m["rate"])
                            )
                            if "start_time" in m:
                                m_asymmetric[-1]["start_time"] = m["start_time"]
                            if "end_time" in m:
                                m_asymmetric[-1]["end_time"] = m["end_time"]
                else:
                    # all ms in m_list are asymmetric
                    for m in m_list:
                        m_asymmetric.append(
                            dict(source=source, dest=dest, rate=m["rate"])
                        )
                        if "start_time" in m:
                            m_asymmetric[-1]["start_time"] = m["start_time"]
                        if "end_time" in m:
                            m_asymmetric[-1]["end_time"] = m["end_time"]
            migrations_out = {}
            if len(m_symmetric) > 0:
                migrations_out["symmetric"] = m_symmetric
            if len(m_asymmetric) > 0:
                migrations_out["asymmetric"] = m_asymmetric
            if len(migrations_out) > 0:
                d.update(migrations=migrations_out)

        if len(self.pulses) > 0:
            d.update(pulses=[attr.asdict(pulse) for pulse in self.pulses])

        return d
