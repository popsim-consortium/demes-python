import numpy as np

from . import demes


def island(size, migration_rate):
    """
    Returns a :class:`.Graph` object representing a collection of
    demes with specified sizes with symmetric migration between
    each pair of demes at the specified rate.

    :param array_like size: the ``start_size`` value for each
        of :class:`.Deme` in the returned model. The length
        of the array corresponds to the number of demes.
    :param float migration_rate: The migration rate between each pair of
        demes.
    :return: A Graph object representing this model.
    :rtype: .Graph
    """
    graph = demes.Graph(description="Island model", time_units="generations")
    for j, start_size in enumerate(size):
        graph.deme(
            id=f"pop_{j}", epochs=[demes.Epoch(start_size=start_size, end_time=0)]
        )
    graph.symmetric_migration(
        demes=[deme.id for deme in graph.demes], rate=migration_rate
    )
    return graph


def stepping_stone(size, migration_rate, boundaries=False):
    """
    Returns a :class:`.Graph` object representing a collection of
    demes with the specified population sizes and in which adjacent
    demes exchange migrants at the specified rate.

    .. note:: The current implementation only supports a one-dimensional
        stepping stone model, but higher dimensions could also be supported.
        Please open an issue on GitHub if this feature would be useful to you.

    :param array_like size: the size of each deme in the returned model.
        The length of the array corresponds to the number of demes.
    :param float migration_rate: The migration rate between adjacent pairs
        of demes.
    :param bool boundaries: If True the stepping stone model has boundary
        conditions imposed so that demes at either end of the chain do
        not exchange migrants. If False (the default), the set of
        populations is "circular" and migration takes place between the
        terminal demes.
    :return: A Graph object representing this model.
    :rtype: .Graph
    """
    size = np.array(size, dtype=np.float64)
    if len(size.shape) > 1:
        raise ValueError(
            "Only 1D stepping stone models currently supported. Please open "
            "an issue on GitHub if you would like 2D (or more) models"
        )
    K = size.shape[0]
    graph = demes.Graph(description="1D stepping model", time_units="generations")
    for j, start_size in enumerate(size):
        graph.deme(
            id=f"pop_{j}", epochs=[demes.Epoch(start_size=start_size, end_time=0)]
        )

    for j in range(K - 1):
        graph.symmetric_migration(
            demes=[graph.demes[j].id, graph.demes[j + 1].id], rate=migration_rate
        )
    if K > 1 and not boundaries:
        graph.symmetric_migration(
            demes=[graph.demes[0].id, graph.demes[-1].id], rate=migration_rate
        )
    return graph
