from typing import List, Union, Optional, Dict, MutableMapping, Any, Set
import itertools
import math
import numbers
import copy
import operator
import warnings

import attr

Number = Union[int, float]
Name = str
Time = Number
Size = Number
Rate = float
Proportion = float

_ISCLOSE_REL_TOL = 1e-9
_ISCLOSE_ABS_TOL = 1e-12

# Validator functions.


def int_or_float(self, attribute, value):
    if (
        not isinstance(value, numbers.Real) and not hasattr(value, "__float__")
    ) or value != value:  # type-agnostic test for NaN
        raise TypeError(f"{attribute.name} must be a number")


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


def nonzero_len(self, attribute, value):
    if len(value) == 0:
        if isinstance(value, str):
            raise ValueError(f"{attribute.name} must be a non-empty string")
        else:
            raise ValueError(f"{attribute.name} must have non-zero length")


def valid_deme_name(self, attribute, value):
    if not value.isidentifier():
        raise ValueError(
            "Invalid deme name `{self.name}`. Names must be valid python identifiers. "
            "We recommend choosing a name that starts with a letter or "
            "underscore, and is followed by one or more letters, numbers, "
            "or underscores."
        )


def isclose_deme_proportions(
    a_names: List[Name],
    a_proportions: List[Proportion],
    b_names: List[Name],
    b_proportions: List[Proportion],
    *,
    rel_tol=_ISCLOSE_REL_TOL,
    abs_tol=_ISCLOSE_ABS_TOL,
) -> bool:
    """
    Returns true if (a_names, a_proportions) and (b_names, b_proportions)
    are semantically equivalent. The order of names is ignored, and proportions
    are checked for numerical closeness.
    """
    if len(a_names) != len(b_names) or len(a_proportions) != len(b_proportions):
        return False
    a = sorted(zip(a_names, a_proportions), key=operator.itemgetter(0))
    b = sorted(zip(b_names, b_proportions), key=operator.itemgetter(0))
    for (a_id, a_proportion), (b_id, b_proportion) in zip(a, b):
        if a_id != b_id or not math.isclose(
            a_proportion, b_proportion, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return False
    return True


def validate_item(name, value, required_type, scope):
    if not isinstance(value, required_type):
        raise TypeError(
            f"{scope}: field '{name}' must be a {required_type}; "
            f"current type is {type(value)}."
        )


# We need to use this trick because None is a meaninful input value for these
# pop_x functions.
NO_DEFAULT = object()


def pop_item(data, name, *, required_type, default=NO_DEFAULT, scope=""):
    if name in data:
        value = data.pop(name)
        validate_item(name, value, required_type, scope=scope)
    else:
        if default is NO_DEFAULT:
            raise KeyError(f"{scope}: required field '{name}' not found")
        value = default
    return value


def pop_list(data, name, default=NO_DEFAULT, required_type=None, scope=""):
    value = pop_item(data, name, default=default, required_type=list)
    if required_type is not None and default is not None:
        for item in value:
            validate_item(name, item, required_type, scope)
    return value


def pop_object(data, name, default=NO_DEFAULT, scope=""):
    return pop_item(
        data, name, default=default, required_type=MutableMapping, scope=scope
    )


def check_allowed(data, allowed_fields, scope):
    for key in data.keys():
        if key not in allowed_fields:
            raise KeyError(
                f"{scope}: unexpected field: '{key}'. "
                f"Allowed fields are: {allowed_fields}"
            )


def insert_defaults(data, defaults):
    for key, value in defaults.items():
        if key not in data:
            data[key] = value


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Epoch:
    """
    Population size parameters for a deme in a specified time period.
    Times follow the forwards-in-time convention (time values increase
    from the present towards the past). The start time of the epoch is
    the more ancient time, and the end time is more recent, so that the
    start time must be greater than the end time

    :ivar float start_time: The start time of the epoch.
    :ivar float ~.end_time: The end time of the epoch (must be specified).
    :ivar float start_size: Population size at ``start_time``.
    :ivar float end_size: Population size at ``end_time``.
        If ``start_size != end_size``, the population size changes
        monotonically between the start and end times.
    :ivar str size_function: The size change function. This is either
        ``constant`` or ``exponential``, though it is possible that
        additional values will be added in the future.

         * ``constant``: the deme's size does not change over the epoch.
         * ``exponential``: the deme's size changes exponentially from
           ``start_size`` to ``end_size`` over the epoch.
           If :math:`t` is a time within the span of the epoch,
           the deme size :math:`N` at :math:`t` can be calculated as:

           .. code::

               dt = (epoch.start_time - t) / epoch.time_span
                r = math.log(epoch.end_size / epoch.start_size)
                N = epoch.start_size * math.exp(r * dt)

        .. warning::

            Do not assume an exponentially changing size just because
            ``start_size != end_size``. For forwards compatibility,
            applications should always check the ``size_function``.

    :ivar float selfing_rate: The selfing rate for this epoch.
    :ivar float cloning_rate: The cloning rate for this epoch.
    """

    start_time: Time = attr.ib(validator=[int_or_float, non_negative])
    end_time: Time = attr.ib(validator=[int_or_float, non_negative, finite])
    start_size: Size = attr.ib(validator=[int_or_float, positive, finite])
    end_size: Size = attr.ib(validator=[int_or_float, positive, finite])
    size_function: str = attr.ib(
        validator=attr.validators.in_(["constant", "exponential"])
    )
    selfing_rate: Proportion = attr.ib(
        default=0, validator=[int_or_float, unit_interval]
    )
    cloning_rate: Proportion = attr.ib(
        default=0, validator=[int_or_float, unit_interval]
    )

    def __attrs_post_init__(self):
        if self.start_time <= self.end_time:
            raise ValueError("must have start_time > end_time")
        if math.isinf(self.start_time) and self.start_size != self.end_size:
            raise ValueError("if start time is inf, must be a constant size epoch")
        if self.size_function == "constant" and self.start_size != self.end_size:
            raise ValueError("start_size != end_size, but size_function is constant")
        if self.selfing_rate + self.cloning_rate > 1:
            raise ValueError("must have selfing_rate + cloning_rate <= 1")

    @property
    def time_span(self):
        """
        The time span of the epoch.
        """
        return self.start_time - self.end_time

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``start_time``, ``end_time``, ``start_size``, ``end_size``,
        ``size_function``, ``selfing_rate``, ``cloning_rate``.

        :param other: The epoch to compare against.
        :type other: :class:`.Epoch`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other epoch is not instance of {self.__class__} type."
        assert math.isclose(
            self.start_time, other.start_time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for start_time {self.start_time} != {other.start_time} (other)."
        assert math.isclose(
            self.end_time, other.end_time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for end_time {self.end_time} != {other.end_time} (other)."
        assert math.isclose(
            self.start_size, other.start_size, rel_tol=rel_tol, abs_tol=abs_tol
        ), (
            f"Failed for start_size "
            f"{self.start_size} != {other.start_size} (other)."
        )
        assert math.isclose(
            self.end_size, other.end_size, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for end_size {self.end_size} != {other.end_size} (other)."
        assert self.size_function == other.size_function
        assert math.isclose(
            self.selfing_rate, other.selfing_rate, rel_tol=rel_tol, abs_tol=abs_tol
        ), (
            f"Failed for selfing_rate "
            f"{self.selfing_rate} != {other.selfing_rate} (other)."
        )
        assert math.isclose(
            self.cloning_rate, other.cloning_rate, rel_tol=rel_tol, abs_tol=abs_tol
        ), (
            f"Failed for cloning_rate "
            f"{self.cloning_rate} != {other.cloning_rate} (other)."
        )

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the epoch and ``other`` epoch implement essentially
        the same epoch. For more information see :meth:`assert_close`.

        :param other: The epoch to compare against.
        :type other: :class:`.Epoch`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float


        :return: True if the two epochs are equivalent, False otherwise.
        :rtype: bool
        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class AsymmetricMigration:
    """
    Parameters for continuous asymmetric migration.
    The source and destination demes follow the forwards-in-time convention,
    where migrants are born in the source deme and (potentially) have children
    in the dest deme.

    :ivar str source: The source deme for asymmetric migration.
    :ivar str dest: The destination deme for asymmetric migration.
    :ivar float start_time: The time at which the migration rate is activated.
    :ivar float ~.end_time: The time at which the migration rate is deactivated.
    :ivar float rate: The rate of migration per generation.
    """

    source: Name = attr.ib(
        validator=[attr.validators.instance_of(str), valid_deme_name]
    )
    dest: Name = attr.ib(validator=[attr.validators.instance_of(str), valid_deme_name])
    start_time: Time = attr.ib(validator=[int_or_float, non_negative])
    end_time: Time = attr.ib(validator=[int_or_float, non_negative, finite])
    rate: Rate = attr.ib(validator=[int_or_float, unit_interval])

    def __attrs_post_init__(self):
        if self.source == self.dest:
            raise ValueError("source and dest cannot be the same deme")
        if not (self.start_time > self.end_time):
            raise ValueError("must have start_time > end_time")

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``source``, ``dest``, ``start_time``, ``end_time``, ``rate``.

        :param AsymmetricMigration other: The migration to compare against.
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other migration is not instance of {self.__class__} type."
        assert self.source == other.source
        assert self.dest == other.dest
        assert math.isclose(
            self.start_time, other.start_time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for start_time {self.start_time} != {other.start_time} (other)."
        assert math.isclose(
            self.end_time, other.end_time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for end_time {self.end_time} != {other.end_time} (other)."
        assert math.isclose(
            self.rate, other.rate, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for rate {self.rate} != {other.rate} (other)."

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the migration is equal to the ``other`` migration.
        For more information see :meth:`assert_close`.

        :param AsymmetricMigration other: The migration to compare against.
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two epochs are equivalent, False otherwise.
        :rtype: bool

        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Pulse:
    """
    Parameters for a pulse of migration from one deme to another.
    Source and destination demes follow the forwards-in-time convention,
    of migrations born in the source deme having children in the dest
    deme.

    :ivar str source: The source deme.
    :ivar str dest: The destination deme.
    :ivar float ~.time: The time of migration.
    :ivar float proportion: At the instant after migration, this is the
        proportion of individuals in the destination deme made up of
        individuals from the source deme.
    """

    source: Name = attr.ib(
        validator=[attr.validators.instance_of(str), valid_deme_name]
    )
    dest: Name = attr.ib(validator=[attr.validators.instance_of(str), valid_deme_name])
    time: Time = attr.ib(validator=[int_or_float, positive, finite])
    proportion: Proportion = attr.ib(validator=[int_or_float, unit_interval])

    def __attrs_post_init__(self):
        if self.source == self.dest:
            raise ValueError("source and dest cannot be the same deme")

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``source``, ``dest``, ``time``, ``proportion``.

        :param other: The pulse to compare against.
        :type other: :class:`.Pulse`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other pulse is not instance of {self.__class__} type."
        assert self.source == other.source
        assert self.dest == other.dest
        assert math.isclose(
            self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for time {self.time} != {other.time} (other)."
        assert math.isclose(
            self.proportion, other.proportion, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for proportion {self.proportion} != {other.proportion} (other)."

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the pulse is equal to the ``other`` pulse.
        For more information see :meth:`assert_close`.

        :param other: The pulse to compare against.
        :type other: :class:`.Pulse`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two pulses are equivalent, False otherwise.
        :rtype: bool

        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Split:
    """
    Parameters for a split event, in which a deme ends at a given time and
    contributes ancestry to an arbitrary number of descendant demes. Note
    that there could be just a single descendant deme, in which case ``split``
    is a bit of a misnomer...

    :ivar str parent: The parental deme.
    :ivar list[str] children: A list of descendant demes.
    :ivar float ~.time: The split time.
    """

    parent: Name = attr.ib(
        validator=[attr.validators.instance_of(str), valid_deme_name]
    )
    children: List[Name] = attr.ib(
        validator=attr.validators.and_(
            attr.validators.deep_iterable(
                member_validator=attr.validators.and_(
                    attr.validators.instance_of(str), valid_deme_name
                ),
                iterable_validator=attr.validators.instance_of(list),
            ),
            nonzero_len,
        )
    )
    time: Time = attr.ib(validator=[int_or_float, non_negative, finite])

    def __attrs_post_init__(self):
        if self.parent in self.children:
            raise ValueError("child and parent cannot be the same deme")
        if len(set(self.children)) != len(self.children):
            raise ValueError("cannot repeat children in split")

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``parent``, ``children``, ``time``.

        :param other: The split to compare against.
        :type other: :class:`.Split`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other split is not instance of {self.__class__} type."
        assert self.parent == other.parent
        assert sorted(self.children) == sorted(other.children)
        assert math.isclose(
            self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for time {self.time} != {other.time} (other)."
        return True

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the split is equal to the ``other`` split.
        Compares values of the following attributes:
        ``parent``, ``children``, ``time``.

        :param other: The split to compare against.
        :type other: :class:`.Split`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two splits are equivalent, False otherwise.
        :rtype: bool
        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Branch:
    """
    Parameters for a branch event, where a new deme branches off from a parental
    deme. The parental deme need not end at that time.

    :ivar str parent: The parental deme.
    :ivar str child: The descendant deme.
    :ivar float ~.time: The branch time.
    """

    parent: Name = attr.ib(
        validator=[attr.validators.instance_of(str), valid_deme_name]
    )
    child: Name = attr.ib(validator=[attr.validators.instance_of(str), valid_deme_name])
    time: Time = attr.ib(validator=[int_or_float, non_negative, finite])

    def __attrs_post_init__(self):
        if self.child == self.parent:
            raise ValueError("child and parent cannot be the same deme")

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``parent``, ``child``, ``time``.

        :param other: The branch to compare against.
        :type other: :class:`.Branch`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"failed as other branch is not instance of {self.__class__} type."
        assert self.parent == other.parent
        assert self.child == other.child
        assert math.isclose(
            self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for time {self.time} != {other.time} (other)."

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the branch is equal to the ``other`` branch.
        For more information see :meth:`assert_close`.

        :param other: The branch to compare against.
        :type other: :class:`.Branch`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two branches are equivalent, False otherwise.
        :rtype: bool

        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Merge:
    """
    Parameters for a merge event, in which two or more demes end at some time and
    contribute to a descendant deme.

    :ivar list[str] parents: A list of parental demes.
    :ivar list[float] proportions: A list of ancestry proportions,
        in order of ``parents``.
    :ivar str child: The descendant deme.
    :ivar float ~.time: The merge time.
    """

    parents: List[Name] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.and_(
                attr.validators.instance_of(str), valid_deme_name
            ),
            iterable_validator=attr.validators.instance_of(list),
        )
    )
    proportions: List[Proportion] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=int_or_float,
            iterable_validator=attr.validators.instance_of(list),
        )
    )
    child: Name = attr.ib(validator=[attr.validators.instance_of(str), valid_deme_name])
    time: Time = attr.ib(validator=[int_or_float, non_negative, finite])

    @proportions.validator
    def _check_proportions(self, attribute, _value):
        if len(self.proportions) > 0 and not math.isclose(sum(self.proportions), 1.0):
            raise ValueError("proportions must sum to 1.0")
        for proportion in self.proportions:
            unit_interval(self, attribute, proportion)
            positive(self, attribute, proportion)

    def __attrs_post_init__(self):
        if len(self.parents) < 2:
            raise ValueError("merge must involve at least two ancestors")
        if len(self.parents) != len(self.proportions):
            raise ValueError("parents and proportions must have same length")
        if self.child in self.parents:
            raise ValueError("merged deme cannot be its own ancestor")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("cannot repeat parents in merge")

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``parents``, ``proportions``, ``child``, ``time``.

        :param other: The merge to compare against.
        :type other: :class:`.Merge`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other merge is not instance of {self.__class__} type."
        assert isclose_deme_proportions(
            self.parents,
            self.proportions,
            other.parents,
            other.proportions,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ), (
            f"Parents or corresponding proportions are different: "
            f"parents: {self.parents}, {other.parents} (other), "
            f"proportions: {self.proportions}, {other.proportions} (other)."
        )
        assert self.child == other.child
        assert math.isclose(
            self.time, other.time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for time {self.time} != {other.time} (other)."

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the merge is equal to the ``other`` merge.
        For more information see :meth:`assert_close`.

        :param other: The merge to compare against.
        :type other: :class:`.Merge`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two merges are equivalent, False otherwise.
        :rtype: bool

        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Admix:
    """
    Parameters for an admixture event, where two or more demes contribute ancestry
    to a new deme.

    :ivar list[str] parents: A list of source demes.
    :ivar list[float] proportions: A list of ancestry proportions,
        in order of ``parents``.
    :ivar str child: The admixed deme.
    :ivar float ~.time: The admixture time.
    """

    parents: List[Name] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.and_(
                attr.validators.instance_of(str), valid_deme_name
            ),
            iterable_validator=attr.validators.instance_of(list),
        )
    )
    proportions: List[Proportion] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=int_or_float,
            iterable_validator=attr.validators.instance_of(list),
        )
    )
    child: Name = attr.ib(validator=[attr.validators.instance_of(str), valid_deme_name])
    time: Time = attr.ib(validator=[int_or_float, non_negative, finite])

    @proportions.validator
    def _check_proportions(self, attribute, _value):
        if len(self.proportions) > 0 and not math.isclose(sum(self.proportions), 1.0):
            raise ValueError("proportions must sum to 1.0")
        for proportion in self.proportions:
            unit_interval(self, attribute, proportion)
            positive(self, attribute, proportion)

    def __attrs_post_init__(self):
        if len(self.parents) < 2:
            raise ValueError("admixture must involve at least two ancestors")
        if len(self.parents) != len(self.proportions):
            raise ValueError("parents and proportions must have same length")
        if self.child in self.parents:
            raise ValueError("admixed deme cannot be its own ancestor")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("cannot repeat parents in admixure")

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following attributes:
        ``parents``, ``proportions``, ``child``, ``time``.

        :param other: The admixture to compare against.
        :type other: :class:`.Admix`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other admixture is not instance of {self.__class__} type."
        assert isclose_deme_proportions(
            self.parents,
            self.proportions,
            other.parents,
            other.proportions,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ), (
            f"Parents or corresponding proportions are different: "
            f"parents: {self.parents}, {other.parents} (other), "
            f"proportions: {self.proportions}, {other.proportions} (other)."
        )
        assert self.child == other.child
        assert math.isclose(
            self.time,
            other.time,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ), f"Failed for time {self.time} != {other.time} (other)."

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the admixture is equal to the ``other`` admixture.
        For more information see :meth:`assert_close`.

        :param other: The admixture to compare against.
        :type other: :class:`.Admix`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two admixtures are equivalent, False otherwise.
        :rtype: bool
        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Deme:
    """
    A collection of individuals that have a common set of population parameters.

    :ivar str name: A conscise string that identifies the deme.
    :ivar str description: A description of the deme. May be ``None``.
    :ivar float start_time: The time at which the deme begins to exist.
    :ivar list[str] ancestors: List of deme names for the deme's ancestors.
        This may be ``None``, indicating the deme has no ancestors.
    :ivar list[float] proportions: If ``ancestors`` is not ``None``,
        this indicates the proportions of ancestry from each ancestor.
        This list has the same length as ``ancestors``, and must sum to 1.
    :ivar list[Epoch] epochs: A list of epochs, which define the population
        size(s) of the deme. The deme must be created with all epochs listed.
    """

    name: Name = attr.ib(validator=[attr.validators.instance_of(str), valid_deme_name])
    description: Optional[str] = attr.ib(
        validator=attr.validators.optional(
            [attr.validators.instance_of(str), nonzero_len]
        )
    )
    start_time: Time = attr.ib(validator=[int_or_float, positive])
    ancestors: List[Name] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.and_(
                attr.validators.instance_of(str), valid_deme_name
            ),
            iterable_validator=attr.validators.instance_of(list),
        )
    )
    proportions: List[Proportion] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=int_or_float,
            iterable_validator=attr.validators.instance_of(list),
        )
    )
    epochs: List[Epoch] = attr.ib(
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(Epoch),
            iterable_validator=attr.validators.instance_of(list),
        )
    )

    @ancestors.validator
    def _check_ancestors(self, _attribute, _value):
        if len(set(self.ancestors)) != len(self.ancestors):
            raise ValueError(
                f"deme {self.name}: duplicate ancestors in {self.ancestors}"
            )
        if self.name in self.ancestors:
            raise ValueError(f"deme {self.name}: deme cannot be its own ancestor")

    @proportions.validator
    def _check_proportions(self, attribute, _value):
        if len(self.proportions) > 0 and not math.isclose(sum(self.proportions), 1.0):
            raise ValueError(f"deme {self.name}: ancestry proportions must sum to 1.0")
        for proportion in self.proportions:
            unit_interval(self, attribute, proportion)
            positive(self, attribute, proportion)

    @epochs.validator
    def _check_epochs(self, _attribute, _value):
        # check epoch times align
        for i, epoch in enumerate(self.epochs):
            if i > 0:
                if self.epochs[i - 1].end_time != epoch.start_time:
                    raise ValueError(
                        f"deme {self.name}: "
                        f"epoch[{i}].start_time != epoch[{i}-1].end_time"
                    )

    def __attrs_post_init__(self):
        # We check the lengths of ancestors and proportions match
        # after the validators have confirmed that these are indeed lists.
        if len(self.ancestors) != len(self.proportions):
            raise ValueError(
                f"deme {self.name}: ancestors and proportions have different lengths"
            )

    def _add_epoch(
        self,
        *,
        end_time,
        start_size=None,
        end_size=None,
        size_function=None,
        selfing_rate=0,
        cloning_rate=0,
    ):
        if len(self.epochs) == 0:
            start_time = self.start_time
            # The first epoch is special.
            if start_size is None and end_size is None:
                raise KeyError(
                    f"deme {self.name}: first epoch must have start_size or end_size"
                )
            if start_size is None:
                start_size = end_size
            if end_size is None:
                end_size = start_size

        else:
            start_time = self.epochs[-1].end_time
            # Set size based on previous epoch.
            if start_size is None:
                start_size = self.epochs[-1].end_size
            if end_size is None:
                end_size = start_size

        if size_function is None:
            if start_size == end_size:
                size_function = "constant"
            else:
                size_function = "exponential"

        try:
            epoch = Epoch(
                start_time=start_time,
                end_time=end_time,
                start_size=start_size,
                end_size=end_size,
                size_function=size_function,
                selfing_rate=selfing_rate,
                cloning_rate=cloning_rate,
            )
        except (TypeError, ValueError) as e:
            raise e.__class__(
                f"deme {self.name}: epoch[{len(self.epochs)}]: invalid epoch"
            ) from e
        self.epochs.append(epoch)

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Compares values of the following objects:
        ``name``, ``ancestors``, ``proportions``, epochs.

        :param other: The deme to compare against.
        :type other: :class:`.Deme`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float
        """
        assert (
            self.__class__ is other.__class__
        ), f"Failed as other deme is not instance of {self.__class__} type."
        assert self.name == other.name
        assert math.isclose(
            self.start_time, other.start_time, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Failed for start_time {self.start_time} != {other.start_time} (other)."
        assert isclose_deme_proportions(
            self.ancestors,
            self.proportions,
            other.ancestors,
            other.proportions,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ), (
            f"Ancestors or corresponding proportions are different: "
            f"ancestors: {self.ancestors}, {other.ancestors} (other), "
            f"proportions: {self.proportions}, {other.proportions} (other)."
        )
        for i, (e1, e2) in enumerate(zip(self.epochs, other.epochs)):
            try:
                e1.assert_close(e2, rel_tol=rel_tol, abs_tol=abs_tol)
            except AssertionError as e:
                raise AssertionError(f"Failed for epochs (number {i})") from e

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the deme is equal to the ``other`` deme.
        For more information see :meth:`assert_close`.

        :param other: The deme to compare against.
        :type other: :class:`.Deme`
        :param ret_tol: The relative tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`
        :type ret_tol: float
        :param abs_tol: The absolute tolerance permitted for numerical
                        comparisons. See documentation for :func:`math.isclose`.
        :type abs_tol: float

        :return: True if the two demes are equivalent, False otherwise.
        :rtype: bool
        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False

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


@attr.s(auto_attribs=True, kw_only=True, slots=True)
class Graph:
    """
    The Graph class provides a high-level API for working with a demographic
    model. A Graph object matches Demes' data model, with a small number of
    additional redundant attributes that make the Graph a more convenient
    object to use when inspecting a model's properties.

    :ivar str description: A human readable description of the demography.
        May be ``None``.
    :ivar str time_units: The units of time used for the demography. This is
        commonly ``years`` or ``generations``, but can be any string.
        This field is intended to be useful for documenting a demography,
        but the actual value provided here should not be relied upon.
    :ivar float generation_time: The generation time of demes, in units given
        by the ``time_units`` parameter. Concretely, dividing all times
        by ``generation_time`` will convert the graph to have time
        units in generations.  If ``generation_time`` is ``None``, the units
        are assumed to be in generations already.
        See also: :meth:`.in_generations`.
    :ivar list[str] doi: If the graph describes a published demography,
        the DOI(s) should be be given here as a list.
    :ivar list[Deme] demes: The demes in the demography.
    :ivar list[AsymmetricMigration] migrations: The continuous migrations for
        the demographic model.
    :ivar list[Pulse] pulses: The migration pulses for the demography.
    """

    description: Optional[str] = attr.ib(
        default=None,
        validator=attr.validators.optional(
            [attr.validators.instance_of(str), nonzero_len]
        ),
    )
    time_units: str = attr.ib(validator=[attr.validators.instance_of(str), nonzero_len])
    generation_time: Optional[Time] = attr.ib(
        default=None,
        validator=attr.validators.optional([int_or_float, positive, finite]),
    )
    doi: List[str] = attr.ib(
        factory=list,
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.and_(
                attr.validators.instance_of(str), nonzero_len
            ),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )
    demes: List[Deme] = attr.ib(factory=list, init=False)
    migrations: List[AsymmetricMigration] = attr.ib(factory=list, init=False)
    pulses: List[Pulse] = attr.ib(factory=list, init=False)

    _deme_map: Dict[Name, Deme] = attr.ib(factory=dict, init=False)

    def __attrs_post_init__(self):
        if self.time_units != "generations" and self.generation_time is None:
            raise ValueError(
                'if time_units!="generations", generation_time must be specified'
            )

    def __getitem__(self, deme_name):
        """
        Return the :class:`.Deme` with the specified name.
        """
        return self._deme_map[deme_name]

    def __contains__(self, deme_name):
        """
        Check if the graph contains a deme with the specified name.
        """
        return deme_name in self._deme_map

    def assert_close(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ):
        """
        Raises AssertionError if the object is not equal to ``other``,
        up to a numerical tolerance.
        Furthermore, the following implementation details are ignored during
        the comparison:

            - The graphs' ``description`` and ``doi`` attributes.
            - The order in which ``migrations`` were specified.
            - The order in which admixture ``pulses`` were specified.
            - The order in which ``demes`` were specified.
            - The order in which a deme's ``ancestors`` were specified.

        :param other: The graph to compare against.
        :type other: :class:`.Graph`
        :param float rel_tol: The relative tolerance permitted for numerical
            comparisons. See documentation for :func:`math.isclose`.
        :param float abs_tol: The absolute tolerance permitted for numerical
            comparisons. See documentation for :func:`math.isclose`.
        """

        def assert_sorted_eq(aa, bb, *, rel_tol, abs_tol, name):
            # Order-agnostic equality check.
            assert len(aa) == len(bb)
            for (a, b) in zip(sorted(aa), sorted(bb)):
                try:
                    a.assert_close(b, rel_tol=rel_tol, abs_tol=abs_tol)
                except AssertionError as e:
                    if isinstance(a, Deme) and isinstance(b, Deme):
                        raise AssertionError(
                            f"Failed for {name} {a.name} and {b.name}"
                        ) from e
                    raise AssertionError(f"Failed for {name}") from e

        assert (
            self.__class__ is other.__class__
        ), f"Failed as other graph is not instance of {self.__class__} type."
        assert self.time_units == other.time_units
        assert self.generation_time == other.generation_time
        assert_sorted_eq(
            self.demes, other.demes, rel_tol=rel_tol, abs_tol=abs_tol, name="demes"
        )
        assert_sorted_eq(
            self.migrations,
            other.migrations,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            name="migrations",
        )
        assert_sorted_eq(
            self.pulses,
            other.pulses,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            name="pulses",
        )

    def isclose(
        self,
        other,
        *,
        rel_tol=_ISCLOSE_REL_TOL,
        abs_tol=_ISCLOSE_ABS_TOL,
    ) -> bool:
        """
        Returns true if the graph and ``other`` implement essentially
        the same demographic model. Numerical values are compared using the
        :func:`math.isclose` function, from which this method takes its name.
        Furthermore, the following implementation details are ignored during
        the comparison:

            - The graphs' ``description`` and ``doi`` attributes.
            - The order in which ``migrations`` were specified.
            - The order in which admixture ``pulses`` were specified.
            - The order in which ``demes`` were specified.
            - The order in which a deme's ``ancestors`` were specified.

        :param other: The graph to compare against.
        :type other: :class:`.Graph`
        :param float rel_tol: The relative tolerance permitted for numerical
            comparisons. See documentation for :func:`math.isclose`.
        :param float abs_tol: The absolute tolerance permitted for numerical
            comparisons. See documentation for :func:`math.isclose`.
        :return: True if the two graphs implement the same model, False otherwise.
        :rtype: bool
        """
        try:
            self.assert_close(other, rel_tol=rel_tol, abs_tol=abs_tol)
            return True
        except AssertionError:
            return False

    def _add_deme(
        self,
        *,
        name,
        description=None,
        ancestors=None,
        proportions=None,
        start_time=None,
    ) -> Deme:
        """
        Add a deme to the graph.

        :param str name: A string identifier for the deme.
        :param list[str] ancestors: List of deme names for the deme's ancestors.
            This may be ``None``, indicating the deme has no ancestors.
            If the deme has multiple ancestors, the ``proportions`` parameter
            must also be provided.
        :param list[float] proportions: The ancestry proportions for ``ancestors``.
            This list has the same length as ``ancestors``, and must sum to ``1.0``.
            May be omitted if the deme has only one, or zero, ancestors.
        :param float start_time: The time at which this deme begins existing,
            in units of ``time_units`` before the present.

            - If the deme has zero ancestors, and ``start_time`` is not specified,
              the start time will be set to ``inf``.
            - If the deme has one ancestor, and ``start_time`` is not specified,
              the ``start_time`` will be set to the ancestor's ``end_time``.
            - If the deme has multiple ancestors, the ``start_time`` must be
              provided.

        :return: Newly created deme.
        :rtype: :class:`.Deme`
        """
        # some basic deme property checks
        if name in self:
            raise ValueError(
                f"deme[{len(self.demes)}] {name}: field 'name' must be unique"
            )
        if ancestors is None:
            ancestors = []
        if not isinstance(ancestors, list):
            raise TypeError(
                f"deme[{len(self.demes)}] {name}: field 'ancestors' must be "
                "a list of deme names"
            )
        for ancestor in ancestors:
            if ancestor not in self:
                raise ValueError(
                    f"deme[{len(self.demes)}] {name}: ancestor deme '{ancestor}' "
                    "not found. Note: ancestor demes must be specified before "
                    "their children."
                )
        if proportions is None:
            if len(ancestors) == 1:
                proportions = [1.0]
            else:
                proportions = []

        # set the start time to inf or to the ancestor's end time
        if start_time is None:
            if len(ancestors) > 0:
                if len(ancestors) > 1:
                    raise ValueError(
                        f"deme[{len(self.demes)}] {name}: "
                        "field 'start_time' not found, "
                        "but is required for demes with multiple ancestors"
                    )
                start_time = self[ancestors[0]].end_time
            else:
                start_time = math.inf

        if len(ancestors) == 0 and not math.isinf(start_time):
            raise ValueError(
                f"deme[{len(self.demes)}] {name}: field 'ancestors' not found, "
                "but is required for demes with a finite 'start_time'"
            )

        # check start time is valid wrt ancestor time intervals
        for ancestor in ancestors:
            anc = self[ancestor]
            if not (anc.start_time > start_time >= anc.end_time):
                raise ValueError(
                    f"deme[{len(self.demes)}] {name}: start_time={start_time} is "
                    "outside the interval of existence for ancestor "
                    f"'{ancestor}' ({anc.start_time}, {anc.end_time}]"
                )

        deme = Deme(
            name=name,
            description=description,
            ancestors=ancestors,
            proportions=proportions,
            start_time=start_time,
            epochs=[],
        )
        self._deme_map[deme.name] = deme
        self.demes.append(deme)
        return deme

    def _check_time_intersection(self, deme1, deme2, time):
        deme1 = self[deme1]
        deme2 = self[deme2]
        time_lo = max(deme1.end_time, deme2.end_time)
        time_hi = min(deme1.start_time, deme2.start_time)
        if time is not None:
            if not (time_lo <= time <= time_hi):
                raise ValueError(
                    f"time {time} not in interval [{time_lo}, {time_hi}], "
                    f"as defined by the time-intersection of {deme1.name} "
                    f"(start_time={deme1.start_time}, end_time={deme1.end_time}) "
                    f"and {deme2.name} (start_time={deme2.start_time}, "
                    f"end_time={deme2.end_time})."
                )
        return time_lo, time_hi

    def _add_symmetric_migration(
        self, *, demes, rate, start_time=None, end_time=None
    ) -> List[AsymmetricMigration]:
        """
        Add continuous symmetric migrations between all pairs of demes in a list.

        :param list[str] demes: Deme names. Migration is symmetric between all
            pairs of demes in this list.
        :param float rate: The rate of migration per generation.
        :param float start_time: The time at which the migration rate is enabled.
        :param float end_time: The time at which the migration rate is disabled.
        :return: List of newly created migrations.
        :rtype: list[AsymmetricMigration]
        """
        if not isinstance(demes, list) or len(demes) < 2:
            raise ValueError("must specify a list of two or more deme names")
        migrations = []
        for source, dest in itertools.permutations(demes, 2):
            migration = self._add_asymmetric_migration(
                source=source,
                dest=dest,
                rate=rate,
                start_time=start_time,
                end_time=end_time,
            )
            migrations.append(migration)
        return migrations

    def _add_asymmetric_migration(
        self, *, source, dest, rate, start_time=None, end_time=None
    ) -> AsymmetricMigration:
        """
        Add continuous migration from one deme to another.
        Source and destination demes follow the forwards-in-time convention,
        so that the migration rate refers to the movement of individuals from
        the ``source`` deme to the ``dest`` deme.

        :param str source: The name of the source deme.
        :param str dest: The name of the destination deme.
        :param float rate: The rate of migration per generation.
        :param float start_time: The time at which the migration rate is enabled.
            If ``None``, the start time is defined by the earliest time at
            which the demes coexist.
        :param float end_time: The time at which the migration rate is disabled.
            If ``None``, the end time is defined by the latest time at which
            the demes coexist.
        :return: Newly created migration.
        :rtype: :class:`.AsymmetricMigration`
        """
        for deme_name in (source, dest):
            if deme_name not in self:
                raise ValueError(f"{deme_name} not in graph")
        time_lo, time_hi = self._check_time_intersection(source, dest, start_time)
        if start_time is None:
            start_time = time_hi
        else:
            self._check_time_intersection(source, dest, start_time)
        if end_time is None:
            end_time = time_lo
        else:
            self._check_time_intersection(source, dest, end_time)
        migration = AsymmetricMigration(
            source=source,
            dest=dest,
            start_time=start_time,
            end_time=end_time,
            rate=rate,
        )

        self.migrations.append(migration)
        return migration

    def _add_pulse(self, *, source, dest, proportion, time) -> Pulse:
        """
        Add a pulse of migration at a fixed time.
        Source and destination demes follow the forwards-in-time convention.

        :param str source: The name of the source deme.
        :param str dest: The name of the destination deme.
        :param float proportion: At the instant after migration, this is the expected
            proportion of individuals in the destination deme made up of individuals
            from the source deme.
        :param float time: The time at which migrations occur.
        :return: Newly created pulse.
        :rtype: :class:`.Pulse`
        """
        for deme_name in (source, dest):
            if deme_name not in self:
                raise ValueError(f"{deme_name} not in graph")
        self._check_time_intersection(source, dest, time)
        if time == self[dest].end_time:
            raise ValueError(
                f"invalid pulse at time={time}, which is dest={dest}'s end_time"
            )
        if time == self[source].start_time:
            raise ValueError(
                f"invalid pulse at time={time}, which is source={source}'s start_time"
            )

        # We create the new pulse object (which checks for common errors)
        # before checking for edge cases below.
        new_pulse = Pulse(
            source=source,
            dest=dest,
            time=time,
            proportion=proportion,
        )

        # Check for models that have multiple pulses defined at the same time.
        # E.g. chains of pulses like: deme0 -> deme1; deme1 -> deme2,
        # where reversing the order of the pulse definitions changes the
        # interpretation of the model. Such models are valid, but the behaviour
        # may not be what the user expects.
        # See https://github.com/popsim-consortium/demes-python/issues/46
        sources = set()
        dests = set()
        for pulse in self.pulses:
            if pulse.time == time:
                sources.add(pulse.source)
                dests.add(pulse.dest)
        if source in dests or dest in (sources | dests):
            warnings.warn(
                "Multiple pulses are defined for the same deme(s) at time "
                f"{time}. The ancestry proportions after this time will thus "
                "depend on the order in which the pulses have been specified. "
                "To avoid unexpected behaviour, the graph can instead "
                "be structured to introduce a new deme at this time with "
                "the desired ancestry proportions."
            )

        # Check for multiple pulses into dest at the same time that
        # give a sum of proportions > 1.
        proportion_sum = proportion
        for pulse in self.pulses:
            if dest == pulse.dest and pulse.time == time:
                proportion_sum += pulse.proportion
        if proportion_sum > 1:
            raise ValueError(
                f"sum of pulse proportions > 1 for dest={dest} at time={time}"
            )

        self.pulses.append(new_pulse)
        return new_pulse

    def _migration_matrices(self):
        """
        Return a list of migration matrices, and a list of end times that
        partition them. The start time for the first matrix is inf.
        """
        uniq_times = set(migration.start_time for migration in self.migrations)
        uniq_times.update(migration.end_time for migration in self.migrations)
        end_times = sorted(uniq_times, reverse=True)[1:]
        n = len(self.demes)
        mm_list = [[[0] * n for _ in range(n)] for _ in range(len(end_times))]
        deme_id = {deme.name: j for j, deme in enumerate(self.demes)}
        for migration in self.migrations:
            start_time = math.inf
            for k, end_time in enumerate(end_times):
                if start_time <= migration.end_time:
                    break
                if end_time < migration.start_time:
                    source_id = deme_id[migration.source]
                    dest_id = deme_id[migration.dest]
                    if mm_list[k][dest_id][source_id] > 0:
                        raise ValueError(
                            "multiple migrations defined for "
                            f"source={migration.source}, dest={migration.dest} "
                            f"between start_time={start_time}, end_time={end_time}"
                        )
                    mm_list[k][dest_id][source_id] = migration.rate
                start_time = end_time
        return mm_list, end_times

    def _check_migration_rates(self):
        """
        Check that the sum of migration ingress rates doesn't exceed 1 for any
        deme in any interval of time.
        """
        start_time = math.inf
        mm_list, end_times = self._migration_matrices()
        for migration_matrix, end_time in zip(mm_list, end_times):
            for j, row in enumerate(migration_matrix):
                row_sum = sum(row)
                if row_sum > 1 and not math.isclose(row_sum, 1):
                    name = self.demes[j].name
                    raise ValueError(
                        f"sum of migration rates into deme {name} is greater "
                        f"than 1 during interval ({start_time}, {end_time}]"
                    )
            start_time = end_time

    def successors(self) -> Dict[Name, List[Name]]:
        """
        Returns the successors (child demes) for all demes in the graph.
        If ``graph`` is a :class:`Graph`, a `NetworkX <https://networkx.org/>`_
        digraph of successors can be obtained as follows.

        .. code::

            import networkx as nx
            succ = nx.from_dict_of_lists(graph.successors(), create_using=nx.DiGraph)

        .. warning::

            The successors do not include information about migrations or pulses.

        :return: A NetworkX compatible dict-of-lists graph of the demes' successors.
        :rtype: dict[str, list[str]]
        """
        succ: Dict[Name, List[Name]] = {}
        for deme_info in self.demes:
            succ.setdefault(deme_info.name, [])
            if deme_info.ancestors is not None:
                for a in deme_info.ancestors:
                    succ.setdefault(a, [])
                    succ[a].append(deme_info.name)
        return succ

    def predecessors(self) -> Dict[Name, List[Name]]:
        """
        Returns the predecessors (ancestors) for all demes in the graph.
        If ``graph`` is a :class:`Graph`, a `NetworkX <https://networkx.org/>`_
        digraph of predecessors can be obtained as follows.

        .. code::

            import networkx as nx
            pred = nx.from_dict_of_lists(graph.predecessors(), create_using=nx.DiGraph)

        .. warning::

            The predecessors do not include information about migrations or pulses.

        :return: A NetworkX compatible dict-of-lists graph of the demes' predecessors.
        :rtype: dict[str, list[str]]
        """
        pred: Dict[Name, List[Name]] = {}
        for deme_info in self.demes:
            pred.setdefault(deme_info.name, [])
            if deme_info.ancestors is not None:
                for a in deme_info.ancestors:
                    pred[deme_info.name].append(a)
        return pred

    def discrete_demographic_events(self) -> Dict[str, List[Any]]:
        """
        Classify each discrete demographic event as one of the following:
        :class:`Pulse`, :class:`Split`, :class:`Branch`, :class:`Merge`,
        or :class:`Admix`.
        If a deme has more than one ancestor, then that deme is created by a
        merger or an admixture event, which are differentiated by end and
        start times of those demes. If a deme has a single predecessor, we check
        whether it is created by a branch (start time != predecessor's end time),
        or split.

        .. note::

            By definition, the discrete demographic events do not include
            migrations, as they are continuous events.

        :return: A dictionary of lists of discrete demographic events.
            The following keys are defined: "pulses", "splits", "branches",
            "mergers", "admixtures", and their values are the corresponding
            lists of :class:`Pulse`, :class:`Split`, :class:`Branch`,
            :class:`Merge`, and :class:`Admix` objects.
        :rtype: dict[str, list]
        """
        demo_events: Dict[str, List[Any]] = {
            "pulses": self.pulses,
            "splits": [],
            "branches": [],
            "mergers": [],
            "admixtures": [],
        }
        splits_to_add: Dict[Name, Set[Name]] = {}
        for c, p in self.predecessors().items():
            if len(p) == 0:
                continue
            elif len(p) == 1:
                if self[c].start_time == self[p[0]].end_time:
                    splits_to_add.setdefault(p[0], set())
                    splits_to_add[p[0]].add(c)
                else:
                    demo_events["branches"].append(
                        Branch(parent=p[0], child=c, time=self[c].start_time)
                    )
            else:
                time_aligned = True
                for deme_from in p:
                    if self[c].start_time != self[deme_from].end_time:
                        time_aligned = False
                if time_aligned is True:
                    demo_events["mergers"].append(
                        Merge(
                            parents=self[c].ancestors,
                            proportions=self[c].proportions,
                            child=c,
                            time=self[c].start_time,
                        )
                    )
                else:
                    demo_events["admixtures"].append(
                        Admix(
                            parents=self[c].ancestors,
                            proportions=self[c].proportions,
                            child=c,
                            time=self[c].start_time,
                        )
                    )
        for deme_from, demes_to in splits_to_add.items():
            demo_events["splits"].append(
                Split(
                    parent=deme_from,
                    children=list(demes_to),
                    time=self[deme_from].end_time,
                )
            )
        return demo_events

    def in_generations(self) -> "Graph":
        """
        Return a copy of the graph with times in units of generations.
        """
        graph = copy.deepcopy(self)
        generation_time = self.generation_time
        graph.generation_time = None
        if graph.time_units != "generations" and generation_time is not None:
            graph.time_units = "generations"
            for deme in graph.demes:
                deme.start_time /= generation_time
                for epoch in deme.epochs:
                    epoch.start_time /= generation_time
                    epoch.end_time /= generation_time
            for migration in graph.migrations:
                migration.start_time /= generation_time
                migration.end_time /= generation_time
            for pulse in graph.pulses:
                pulse.time /= generation_time
        return graph

    @classmethod
    def fromdict(cls, data: MutableMapping[str, Any]) -> "Graph":
        """
        Return a graph from a dict representation. The inverse of asdict().
        """
        if not isinstance(data, MutableMapping):
            raise TypeError("data is not a dictionary")

        # Don't modify the input data dict.
        data = copy.deepcopy(data)

        check_allowed(
            data,
            [
                "description",
                "time_units",
                "generation_time",
                "defaults",
                "doi",
                "demes",
                "migrations",
                "pulses",
            ],
            "toplevel",
        )

        defaults = pop_object(data, "defaults", {}, scope="toplevel")
        check_allowed(
            defaults,
            ["deme", "migration", "pulse", "epoch"],
            "defaults",
        )

        deme_defaults = pop_object(defaults, "deme", {}, scope="defaults")
        allowed_fields_deme = [
            "description",
            "start_time",
            "ancestors",
            "proportions",
        ]
        allowed_fields_deme_inner = allowed_fields_deme + ["name", "defaults", "epochs"]
        check_allowed(deme_defaults, allowed_fields_deme, "defaults: deme")

        migration_defaults = pop_object(defaults, "migration", {}, scope="defaults")
        allowed_fields_migration = [
            "demes",
            "source",
            "dest",
            "start_time",
            "end_time",
            "rate",
        ]
        check_allowed(
            migration_defaults, allowed_fields_migration, "defaults: migration"
        )

        pulse_defaults = pop_object(defaults, "pulse", {}, scope="defaults")
        allowed_fields_pulse = ["source", "dest", "time", "proportion"]
        check_allowed(pulse_defaults, allowed_fields_pulse, "defaults.pulse")

        # epoch defaults may also be specified within a Deme definition.
        global_epoch_defaults = pop_object(defaults, "epoch", {}, scope="defaults")
        allowed_fields_epoch = [
            "end_time",
            "start_size",
            "end_size",
            "size_function",
            "cloning_rate",
            "selfing_rate",
        ]
        check_allowed(global_epoch_defaults, allowed_fields_epoch, "defaults: epoch")

        if "time_units" not in data:
            raise KeyError("toplevel: required field 'time_units' not found")

        graph = cls(
            description=data.pop("description", None),
            time_units=data.pop("time_units"),
            doi=data.pop("doi", []),
            generation_time=data.pop("generation_time", None),
        )

        for i, deme_data in enumerate(
            pop_list(data, "demes", required_type=MutableMapping, scope="toplevel")
        ):
            if "name" not in deme_data:
                raise KeyError("demes[{i}]: required field 'name' not found")
            deme_name = deme_data.pop("name")
            check_allowed(
                deme_data, allowed_fields_deme_inner, f"demes[{i}] {deme_name}"
            )
            insert_defaults(deme_data, deme_defaults)

            deme = graph._add_deme(
                name=deme_name,
                description=deme_data.pop("description", None),
                start_time=deme_data.pop("start_time", None),
                ancestors=deme_data.pop("ancestors", None),
                proportions=deme_data.pop("proportions", None),
            )

            local_defaults = pop_object(
                deme_data, "defaults", {}, scope=f"demes[{i}] {deme.name}"
            )
            check_allowed(
                local_defaults, ["epoch"], f"demes[{i}] {deme.name}: defaults"
            )
            local_epoch_defaults = pop_object(
                local_defaults, "epoch", {}, scope=f"demes[{i}] {deme.name}: defaults"
            )
            epoch_defaults = global_epoch_defaults.copy()
            epoch_defaults.update(local_epoch_defaults)

            check_allowed(
                epoch_defaults,
                allowed_fields_epoch,
                f"demes[{i}] {deme.name}: defaults: epoch",
            )

            if len(epoch_defaults) == 0 and "epochs" not in deme_data:
                # This condition would be caught downstream, because start_size
                # or end_size are required for the first epoch. But we check
                # here to provide a more informative error message.
                raise KeyError(
                    f"demes[{i}] {deme.name}: required field 'epochs' not found"
                )

            # There is always at least one epoch defined with the default values.
            epochs = pop_list(
                deme_data,
                "epochs",
                [{}],
                required_type=MutableMapping,
                scope=f"demes[{i}] {deme.name}",
            )
            for j, epoch_data in enumerate(epochs):
                check_allowed(
                    epoch_data,
                    allowed_fields_epoch,
                    f"demes[{i}] {deme.name}: epochs[{j}]",
                )
                insert_defaults(epoch_data, epoch_defaults)
                if "end_time" not in epoch_data:
                    if j == len(epochs) - 1:
                        epoch_data["end_time"] = 0
                    else:
                        raise KeyError(
                            f"demes[{i}] {deme.name}: epochs[{j}]: "
                            "required field 'end_time' not found"
                        )
                deme._add_epoch(
                    end_time=epoch_data.pop("end_time"),
                    start_size=epoch_data.pop("start_size", None),
                    end_size=epoch_data.pop("end_size", None),
                    size_function=epoch_data.pop("size_function", None),
                    selfing_rate=epoch_data.pop("selfing_rate", 0),
                    cloning_rate=epoch_data.pop("cloning_rate", 0),
                )

        for i, migration_data in enumerate(
            pop_list(
                data, "migrations", [], required_type=MutableMapping, scope="toplevel"
            )
        ):
            check_allowed(migration_data, allowed_fields_migration, f"migration[{i}]")
            insert_defaults(migration_data, migration_defaults)
            if "rate" not in migration_data:
                raise KeyError(f"migration[{i}]: required field 'rate' not found")
            demes = migration_data.pop("demes", None)
            source = migration_data.pop("source", None)
            dest = migration_data.pop("dest", None)
            if not (
                # symmetric
                (demes is not None and source is None and dest is None)
                # asymmetric
                or (demes is None and source is not None and dest is not None)
            ):
                raise KeyError(
                    f"migration[{i}]: must be symmetric (specify 'demes' list) "
                    "*or* asymmetric (specify both 'source' and 'dest')."
                )
            try:
                if demes is not None:
                    graph._add_symmetric_migration(
                        demes=demes,
                        rate=migration_data.pop("rate"),
                        start_time=migration_data.pop("start_time", None),
                        end_time=migration_data.pop("end_time", None),
                    )
                else:
                    graph._add_asymmetric_migration(
                        source=source,
                        dest=dest,
                        rate=migration_data.pop("rate"),
                        start_time=migration_data.pop("start_time", None),
                        end_time=migration_data.pop("end_time", None),
                    )
            except (TypeError, ValueError) as e:
                raise e.__class__(f"migration[{i}]: invalid migration") from e

        graph._check_migration_rates()

        for i, pulse_data in enumerate(
            pop_list(data, "pulses", [], required_type=MutableMapping, scope="toplevel")
        ):
            check_allowed(pulse_data, allowed_fields_pulse, f"pulse[{i}]")
            insert_defaults(pulse_data, pulse_defaults)
            for field in ("source", "dest", "time", "proportion"):
                if field not in pulse_data:
                    raise KeyError(f"pulse[{i}]: required field '{field}' not found")
            try:
                graph._add_pulse(
                    source=pulse_data.pop("source"),
                    dest=pulse_data.pop("dest"),
                    time=pulse_data.pop("time"),
                    proportion=pulse_data.pop("proportion"),
                )
            except (TypeError, ValueError) as e:
                raise e.__class__(f"pulse[{i}]: invalid pulse") from e

        return graph

    def asdict(self) -> MutableMapping[str, Any]:
        """
        Return a fully-resolved dict representation of the graph.
        """

        def filt(attrib, value):
            return (
                value is not None
                and not (hasattr(value, "__len__") and len(value) == 0)
                and attrib.name != "_deme_map"
            )

        def coerce_numbers(inst, attribute, value):
            # Explicitly convert numeric types to int or float, so that they
            # don't cause problems for the YAML and JSON serialisers.
            # E.g. numpy int32/int64 are part of Python's numeric tower as
            # subclasses of numbers.Integral, similarly numpy's float32/float64
            # are subclasses of numbers.Real. There are yet other numeric types,
            # such as the standard library's decimal.Decimal, which are not part
            # of the numeric tower, but provide a __float__() method.
            if isinstance(value, numbers.Integral):
                value = int(value)
            elif isinstance(value, numbers.Real) or hasattr(value, "__float__"):
                value = float(value)
            return value

        data = attr.asdict(self, filter=filt, value_serializer=coerce_numbers)
        # translate to spec data model
        for deme in data["demes"]:
            for epoch in deme["epochs"]:
                del epoch["start_time"]
        return data

    def asdict_simplified(self) -> MutableMapping[str, Any]:
        """
        Return a simplified dict representation of the graph.
        """

        def simplify_epochs(data):
            """
            Remove epoch start times. Also remove deme start time
            if implied by the deme ancestor(s)'s end time(s).
            """
            for deme in data["demes"]:
                for j, epoch in enumerate(deme["epochs"]):
                    if epoch["size_function"] in ("constant", "exponential"):
                        del epoch["size_function"]
                    if epoch["start_size"] == epoch["end_size"]:
                        del epoch["end_size"]
                    if epoch["selfing_rate"] == 0:
                        del epoch["selfing_rate"]
                    if epoch["cloning_rate"] == 0:
                        del epoch["cloning_rate"]

            for deme in data["demes"]:
                # remove implied start times
                if math.isinf(deme["start_time"]):
                    del deme["start_time"]
                if "ancestors" in deme and len(deme["ancestors"]) == 1:
                    del deme["proportions"]
                    # start time needed for more than 1 ancestor
                    if self[deme["ancestors"][0]].end_time == deme["start_time"]:
                        del deme["start_time"]

        def simplify_migration_rates(data):
            """
            Collapse symmetric migration rates, and remove redundant information
            about start and end times if they are implied by the time overlap
            interval of the demes involved.

            To collapse symmetric migrations, we collect all source/dest migration
            pairs for each set of migration attributes (rate, start_time, end_time),
            and then iteratively check for all-way symmetric migration between all
            demes that are involved in migrations for the given set of migration
            attributes.
            """

            def collapse_demes(pairs):
                all_demes = []
                for pair in pairs:
                    if pair[0] not in all_demes:
                        all_demes.append(pair[0])
                    if pair[1] not in all_demes:
                        all_demes.append(pair[1])
                return all_demes

            symmetric = []
            asymmetric = data["migrations"].copy()
            # first remove start/end times if equal time intersections
            rate_sets = {}
            # keys of rate_types are (rate, start_time, end_time)
            for migration in data["migrations"]:
                source = migration["source"]
                dest = migration["dest"]
                time_hi = min(self[source].start_time, self[dest].start_time)
                time_lo = max(self[dest].end_time, self[dest].end_time)
                if migration["end_time"] == time_lo:
                    del migration["end_time"]
                if migration["start_time"] == time_hi:
                    del migration["start_time"]
                k = tuple(
                    migration.get(key) for key in ("rate", "start_time", "end_time")
                )
                rate_sets.setdefault(k, [])
                rate_sets[k].append((source, dest))

            for k, pairs in rate_sets.items():
                if len(pairs) == 1:
                    continue
                # list of all demes that are source or dest in this rate set
                all_demes = collapse_demes(pairs)

                # we check all possible sets of n-way symmetric migration
                i = len(all_demes)
                while len(all_demes) >= 2 and i >= 2:
                    # loop through each possible set for a given set size i
                    compress_demes = False
                    for deme_set in itertools.combinations(all_demes, i):
                        # check if all (source, dest) pairs exist in pairs of migration
                        all_present = True
                        for deme_pair in itertools.permutations(deme_set, 2):
                            if deme_pair not in pairs:
                                all_present = False
                                break
                        # if they do all exist
                        if all_present:
                            compress_demes = True
                            # remove from asymmetric list
                            for deme_pair in itertools.permutations(deme_set, 2):
                                mig = {
                                    "source": deme_pair[0],
                                    "dest": deme_pair[1],
                                    "rate": k[0],
                                }
                                if k[1] is not None:
                                    mig["start_time"] = k[1]
                                if k[2] is not None:
                                    mig["end_time"] = k[2]
                                asymmetric.remove(mig)
                                pairs.remove(deme_pair)
                            # add to symmetric list
                            sym_mig = dict(demes=[d for d in deme_set], rate=k[0])
                            if k[1] is not None:
                                sym_mig["start_time"] = k[1]
                            if k[2] is not None:
                                sym_mig["end_time"] = k[2]
                            symmetric.append(sym_mig)
                    # if we found a set of symmetric migrations, compress all_demes
                    if compress_demes:
                        all_demes = collapse_demes(pairs)
                        i = min(i, len(all_demes))
                    # otherwise, check one set size smaller
                    else:
                        i -= 1

            data["migrations"] = symmetric + asymmetric

        data = self.asdict()

        if "migrations" in data:
            simplify_migration_rates(data)
        simplify_epochs(data)

        return data


class Builder:
    """
    The Builder provides a set of convenient methods for incrementally
    constructing a deme graph. The state of the graph is stored internally as a
    dictionary of objects following Demes' data model, and may be converted
    into a fully-resolved :class:`Graph` object using the :meth:`.resolve()` method.

    :ivar dict data: The data dictionary of the graph's current state.
        The objects nested within this dictionary follow Demes' data model,
        as described in the :ref:`spec:sec_ref`.

        .. note::
            Users may freely modify the data dictionary, as long as the data
            model is not violated.
    """

    def __init__(
        self,
        *,
        description: str = None,
        time_units: str = "generations",
        generation_time: float = None,
        doi: list = None,
        defaults: dict = None,
    ):
        """
        :param str description: A human readable description of the demography.
            May be ``None``.
        :param str time_units: The units of time used for the demography. This is
            commonly ``years`` or ``generations``, but can be any string.
        :param float generation_time: The generation time of demes, in units given
            by the ``time_units`` parameter. Concretely, dividing all times
            by ``generation_time`` will convert the graph to have time
            units in generations.  If ``generation_time`` is ``None``, the units
            are assumed to be in generations already.
        :param doi: If the graph describes a published demography, the DOI(s)
            should be be given here as a list.
        :type doi: list[str]
        """
        self.data: MutableMapping[str, Any] = dict(time_units=time_units)
        if description is not None:
            self.data["description"] = description
        if generation_time is not None:
            self.data["generation_time"] = generation_time
        if doi is not None:
            self.data["doi"] = doi
        if defaults is not None:
            self.data["defaults"] = defaults

    def add_deme(
        self,
        name: str,
        *,
        description: str = None,
        ancestors: list = None,
        proportions: list = None,
        start_time: float = None,
        epochs: list = None,
        defaults: dict = None,
    ):
        """
        Add a deme. Ancestor demes must be added before their children.

        :param str name: A string identifier for the deme.
        :param str description: A description of the deme. May be ``None``.
        :param list[str] ancestors: List of deme names for the deme's ancestors.
            This may be ``None``, indicating the deme has no ancestors.
        :param list[float] proportions: If ``ancestors`` is not ``None``,
            this indicates the proportions of ancestry from each ancestor.
            This list has the same length as ``ancestors``, and must sum to 1.
        :param float start_time: The deme's start time.
        :param list[dict] epochs: List of epoch dictionaries. Each dictionary
            follows the data model for an epoch.
        """
        deme: MutableMapping[str, Any] = dict(name=name)
        if description is not None:
            deme["description"] = description
        if ancestors is not None:
            deme["ancestors"] = ancestors
        if proportions is not None:
            deme["proportions"] = proportions
        if start_time is not None:
            deme["start_time"] = start_time
        if epochs is not None:
            deme["epochs"] = epochs
        if defaults is not None:
            deme["defaults"] = defaults

        if "demes" not in self.data:
            self.data["demes"] = []
        self.data["demes"].append(deme)

    def add_migration(
        self,
        *,
        rate: float = None,
        # We use a special NO_DEFAULT value here, to distinguish between the user
        # not specifying anything, and specifying the value None (which may be
        # necessary to override a 'defaults' value set in the data dictionary).
        demes: list = NO_DEFAULT,  # type: ignore
        source: str = NO_DEFAULT,  # type: ignore
        dest: str = NO_DEFAULT,  # type: ignore
        start_time: float = None,
        end_time: float = None,
    ):
        """
        Add continuous symmetric migrations between all pairs of demes in a list,
        or alternately, add asymmetric migration from one deme to another.
        Source and destination demes follow the forwards-in-time convention,
        so that the migration rate refers to the movement of individuals from
        the ``source`` deme to the ``dest`` deme.

        :param list[str] demes: list of deme names. Migration is symmetric
            between all pairs of demes in this list. If not specified,
            migration will be asymmetric (and ``source`` and ``dest`` must
            be given).
        :param str source: The name of the source deme.
        :param str dest: The name of the destination deme.
        :param float rate: The rate of migration per generation.
        :param float start_time: The time at which the migration rate is enabled.
            If ``None``, the start time is defined by the earliest time at
            which the demes coexist.
        :param float end_time: The time at which the migration rate is disabled.
            If ``None``, the end time is defined by the latest time at which
            the demes coexist.
        """
        migration: MutableMapping[str, Any] = dict()
        if rate is not None:
            migration["rate"] = rate
        if demes is not NO_DEFAULT:
            migration["demes"] = demes
        if source is not NO_DEFAULT:
            migration["source"] = source
        if dest is not NO_DEFAULT:
            migration["dest"] = dest
        if start_time is not None:
            migration["start_time"] = start_time
        if end_time is not None:
            migration["end_time"] = end_time

        if "migrations" not in self.data:
            self.data["migrations"] = []
        self.data["migrations"].append(migration)

    def add_pulse(
        self,
        *,
        source: str = None,
        dest: str = None,
        proportion: float = None,
        time: float = None,
    ):
        """
        Add a pulse of migration at a fixed time.
        Source and destination demes follow the forwards-in-time convention.

        :param str source: The name of the source deme.
        :param str dest: The name of the destination deme.
        :param float proportion: At the instant after migration, this is the
            expected proportion of individuals in the destination deme made up
            of individuals from the source deme.
        :param float time: The time at which migrations occur.
        """
        pulse: MutableMapping[str, Any] = dict()
        if source is not None:
            pulse["source"] = source
        if dest is not None:
            pulse["dest"] = dest
        if proportion is not None:
            pulse["proportion"] = proportion
        if time is not None:
            pulse["time"] = time

        if "pulses" not in self.data:
            self.data["pulses"] = []
        self.data["pulses"].append(pulse)

    def resolve(self):
        """
        Resolve the data dictionary into a Graph.

        :return: The fully-resolved Graph.
        :rtype: Graph
        """
        return Graph.fromdict(self.data)

    @classmethod
    def fromdict(cls, data: MutableMapping[str, Any]) -> "Builder":
        """
        Make a Builder object from an existing data dictionary.

        :param MutableMapping data: The data dictionary to initialise the
            graph's state. The objects nested within this dictionary must
            follow Demes' data model, as described in the :ref:`spec:sec_ref`.

        :return: The new Builder object.
        :rtype: Builder
        """
        builder = cls()
        builder.data = data
        return builder
