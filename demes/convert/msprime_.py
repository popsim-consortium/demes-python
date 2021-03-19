from typing import List, Mapping, Tuple
import math
import collections
import itertools

import attr
import numpy as np
import msprime

import demes


def to_msprime(graph: demes.Graph):
    """
    Convert a demes graph to an msprime demography.

    :param graph: the demes graph to convert.
    :type graph: :class:`demes.Graph`
    :return: a 3-tuple ``(pc, de, mm)``, where
        ``pc`` is a list of population configurations,
        ``de`` is a list of demographic events, and
        ``mm`` is the initial migration matrix.
    :rtype: (
        list of :class:`msprime.PopulationConfiguration`,
        list of :class:`msprime.DemographicEvent`,
        list of list of float)
    """
    graph = graph.in_generations()
    population_configurations = []
    demographic_events = []
    migration_matrix = [[0.0] * len(graph.demes) for _ in range(len(graph.demes))]
    pop_id = {deme.name: j for j, deme in enumerate(graph.demes)}

    def growth_rate(epoch: demes.Epoch) -> float:
        initial_size = epoch.end_size
        final_size = epoch.start_size
        if initial_size == final_size:
            growth_rate = 0.0
        else:
            if epoch.size_function != "exponential":
                raise ValueError(
                    "Unable to set growth_rate for "
                    f"size_function={epoch.size_function}"
                )
            growth_rate = math.log(initial_size / final_size) / epoch.time_span
        return growth_rate

    # Outside of the existence time for a deme, population size should be zero.
    Ne_invalid = 0.0
    try:
        msprime.PopulationConfiguration(initial_size=Ne_invalid)
    except ValueError:
        # Msprime < 1.0 rejects initial_size=0, so use a small positive value.
        Ne_invalid = 1e-15

    for deme in graph.demes:

        if deme.end_time != 0:
            # If this deme doesn't exist at time=0, invalidate Ne.
            initial_size = Ne_invalid
            _growth_rate = 0.0
        else:
            initial_size = deme.epochs[-1].end_size
            _growth_rate = growth_rate(deme.epochs[-1])
        population_configurations.append(
            msprime.PopulationConfiguration(
                initial_size=initial_size,
                growth_rate=_growth_rate,
                metadata=attr.asdict(deme),
            )
        )

        if deme.ancestors is not None:
            for j, ancestor in enumerate(deme.ancestors):
                p = deme.proportions[j] / sum(deme.proportions[j:])
                demographic_events.append(
                    msprime.MassMigration(
                        time=deme.epochs[0].start_time,
                        source=pop_id[deme.name],
                        dest=pop_id[ancestor],
                        proportion=p,
                    )
                )

        for epoch in reversed(deme.epochs):
            if epoch.end_time != 0:
                # If this isn't the initial epoch, change population size.
                demographic_events.append(
                    msprime.PopulationParametersChange(
                        time=epoch.end_time,
                        initial_size=epoch.end_size,
                        growth_rate=growth_rate(epoch),
                        population_id=pop_id[deme.name],
                    )
                )
            if epoch == deme.epochs[0] and not math.isinf(epoch.start_time):
                # If this deme doesn't exist at time=inf, invalidate Ne when
                # the deme ceases to exist.
                demographic_events.append(
                    msprime.PopulationParametersChange(
                        time=epoch.start_time,
                        initial_size=Ne_invalid,
                        growth_rate=0,
                        population_id=pop_id[deme.name],
                    )
                )

    for pulse in graph.pulses:
        demographic_events.append(
            msprime.MassMigration(
                time=pulse.time,
                source=pop_id[pulse.dest],
                dest=pop_id[pulse.source],
                proportion=pulse.proportion,
            )
        )

    mig_rate_events = []

    def append_migration(dest, source, start_time, end_time, rate):
        if start_time == 0:
            migration_matrix[dest][source] = rate
        else:
            mig_rate_events.append(
                msprime.MigrationRateChange(
                    time=start_time,
                    rate=rate,
                    matrix_index=(dest, source),
                )
            )
        mig_rate_events.append(
            msprime.MigrationRateChange(
                time=end_time, rate=0, matrix_index=(dest, source)
            )
        )

    for migration in reversed(graph.migrations):
        rate = migration.rate
        start_time = migration.end_time
        end_time = migration.start_time
        if isinstance(migration, demes.AsymmetricMigration):
            dest = pop_id[migration.source]
            source = pop_id[migration.dest]
            append_migration(dest, source, start_time, end_time, rate)
        else:
            assert isinstance(migration, demes.SymmetricMigration)
            for x, y in itertools.permutations(migration.demes, 2):
                pop_x = pop_id[x]
                pop_y = pop_id[y]
                append_migration(pop_x, pop_y, start_time, end_time, rate)
                append_migration(pop_y, pop_x, start_time, end_time, rate)

    # Collapse migration rate events in the same generation.
    # This is not strictly needed, but usually results in fewer events.
    mig_rate_events.sort(key=lambda de: de.time)
    prev_mm = np.array(migration_matrix)
    off_diagonal = np.where(np.logical_not(np.eye(prev_mm.shape[0], dtype=bool)))
    for _, g in itertools.groupby(mig_rate_events, lambda e: e.time):
        events = list(g)
        mm = prev_mm.copy()
        for de in events:
            mm[de.matrix_index] = de.rate
        if all(mm[off_diagonal] == events[0].rate):
            demographic_events.append(
                msprime.MigrationRateChange(
                    time=events[0].time,
                    rate=events[0].rate,
                )
            )
        else:
            for j in range(mm.shape[0]):
                for k in range(mm.shape[1]):
                    if j != k and mm[j, k] != prev_mm[j, k]:
                        demographic_events.append(
                            msprime.MigrationRateChange(
                                time=events[0].time, rate=mm[j, k], matrix_index=(j, k)
                            )
                        )
        prev_mm = mm

    demographic_events.sort(key=lambda de: de.time)
    return population_configurations, demographic_events, migration_matrix


def from_msprime(
    population_configurations=None,
    demographic_events=None,
    migration_matrix=None,
    pop_names=None,
) -> demes.Graph:
    """
    Convert an msprime demography into a demes graph.

    :param population_configurations: A list of population configurations.
    :type population_configurations: list of :class:`msprime.PopulationConfiguration`
    :param demographic_events: A list of demographic events.
    :type demographic_events: list of :class:`msprime.DemographicEvent`
    :param migration_matrix: The initial migration matrix.
    :type migration_matrix: list of list of float
    :param pop_names: A list of population names to use.
        If None, the names will be pop0, pop1, ..., popN.
    :param pop_names: list of str
    :return: A demes graph.
    :rtype: :class:`demes.Graph`
    """
    ddb = msprime.DemographyDebugger(
        population_configurations=population_configurations,
        demographic_events=demographic_events,
        migration_matrix=migration_matrix,
    )
    num_pops = ddb.num_populations

    if pop_names is None:
        pop_names = [f"pop{j}" for j in range(num_pops)]
    name = {j: pop_name for j, pop_name in enumerate(pop_names)}

    # We first construct a temporary demes graph, to build ancestor/descendent
    # relationships. At the time of insertion into the temporary graph, we
    # don't have complete information about each deme's life-span or population
    # size(s). So we insert dummy epochs into the temporary graph, and build
    # up the correct `Epoch`s, `Migration`s, and `Pulse`s outside of the graph.
    gtmp: Mapping[str, dict] = {"demes": {}}

    # List of epoch dicts, keyed by deme name.
    epochs: Mapping[str, List[dict]] = collections.defaultdict(list)
    # List of deme.Migration, keyed by (source, dest) indexes
    migrations: Mapping[
        Tuple[int, int], List[demes.AsymmetricMigration]
    ] = collections.defaultdict(list)
    # migration_matrix in the previous ddb epoch
    prev_mm = np.zeros((num_pops, num_pops))
    pulses: List[demes.Pulse] = []

    ddb_epochs = sorted(ddb.epochs, key=lambda e: e.start_time, reverse=True)
    for j, ddb_epoch in enumerate(ddb_epochs):
        mass_migrations = collections.defaultdict(list)
        pop_param_changes = set()
        for de in ddb_epoch.demographic_events:
            if isinstance(de, msprime.MassMigration):
                source = name[de.source]
                dest = name[de.dest]
                mass_migrations[source].append((dest, de.proportion))
            elif isinstance(de, msprime.PopulationParametersChange):
                # Make a note of the population, to create a new Epoch
                # for the next ddb_epoch iteration.
                pop_name = name[de.population]
                if de.initial_size is None or de.initial_size > 1e-15:
                    if pop_name not in gtmp["demes"]:
                        gtmp["demes"][pop_name] = {
                            "ancestors": [],
                            "proportions": [],
                            "epochs": [dict(start_size=1)],
                        }
                    pop_param_changes.add(pop_name)

        for child, anc_list in mass_migrations.items():
            ancestors = []
            proportions: List[float] = []
            for anc, p in anc_list:
                ancestors.append(anc)
                remainder = 1.0 - sum(proportions)
                proportions.append(p * remainder)

            for parent in ancestors:
                if parent not in gtmp["demes"]:
                    gtmp["demes"][parent] = {
                        "ancestors": [],
                        "proportions": [],
                        "epochs": [dict(start_size=1)],
                    }

            if math.isclose(sum(proportions), 1):
                assert child not in gtmp
                gtmp["demes"][child] = {}
                # Set attributes after deme creation, to avoid internal
                # checks about the ancestors' existence time intervals.
                gtmp["demes"][child]["epochs"] = [
                    dict(start_time=ddb_epoch.start_time, start_size=1)
                ]
                gtmp["demes"][child]["ancestors"] = ancestors
                gtmp["demes"][child]["proportions"] = proportions
            else:
                if child not in gtmp:
                    pass
                for source, proportion in zip(ancestors, proportions):
                    # Save pulse for adding to the graph later, once all Epochs
                    # are correctly set.
                    pulses.append(
                        demes.Pulse(
                            source=source,
                            dest=child,
                            proportion=proportion,
                            time=ddb_epoch.start_time,
                        )
                    )

        # properly set population sizes, and extend epochs as required
        for j, pop in enumerate(ddb_epoch.populations):
            if name[j] not in gtmp["demes"]:
                continue

            deme_name = name[j]
            if deme_name not in epochs:
                epochs[deme_name].append(gtmp["demes"][deme_name]["epochs"][0])
            last_epoch = epochs[deme_name][-1]
            last_epoch["end_time"] = ddb_epoch.start_time
            if last_epoch.get("start_time") == ddb_epoch.end_time:
                last_epoch["start_size"] = pop.end_size
            last_epoch["end_size"] = pop.start_size

            if name[j] in pop_param_changes:
                # Add new epoch, to be fixed in the next ddb_epoch iteration.
                epochs[deme_name].append(
                    dict(start_time=ddb_epoch.start_time, end_time=0, start_size=1)
                )

        # Construct per-pair lists of migrations from the migration matrix.
        msp_mm = np.array(ddb_epoch.migration_matrix)
        for j in range(num_pops):
            for k in range(num_pops):
                if j == k:
                    continue
                if prev_mm[j, k] != msp_mm[j, k]:
                    if msp_mm[j, k] != 0:
                        # new Migration
                        m = demes.AsymmetricMigration(
                            source=name[j],
                            dest=name[k],
                            start_time=ddb_epoch.end_time,
                            end_time=ddb_epoch.start_time,
                            rate=msp_mm[j, k],
                        )
                        migrations[(j, k)].append(m)
                else:
                    # extend time span of existing Migration
                    if (j, k) in migrations:
                        m = migrations[j, k][-1]
                        if m.end_time == ddb_epoch.end_time:
                            m.end_time = ddb_epoch.start_time
        prev_mm = msp_mm

    for deme_name in epochs.keys():
        epoch = epochs[deme_name][0]
        start_time = epoch.get("start_time")
        if start_time is None or math.isinf(start_time):
            epochs[deme_name][0] = dict(
                start_time=start_time,
                end_time=epoch["end_time"],
                start_size=epoch["end_size"],
                end_size=epoch["end_size"],
            )

    # Create a fresh demes graph, now that we have complete epoch information
    # for each deme. This also validates consistency between parameters.
    b = demes.Builder(
        description="Converted from msprime demography.",
        time_units="generations",
    )

    for deme_name, deme_dict in gtmp["demes"].items():
        b.add_deme(
            deme_name,
            ancestors=deme_dict["ancestors"],
            proportions=deme_dict["proportions"],
            start_time=epochs[deme_name][0]["start_time"],
            epochs=[
                dict(
                    end_time=epoch["end_time"],
                    start_size=epoch["start_size"],
                    end_size=epoch["end_size"],
                )
                for epoch in epochs[deme_name]
            ],
        )

    for pulse in pulses:
        b.add_pulse(
            source=pulse.source,
            dest=pulse.dest,
            proportion=pulse.proportion,
            time=pulse.time,
        )

    for migration_list in migrations.values():
        for migration in migration_list:
            b.add_migration(
                source=migration.source,
                dest=migration.dest,
                start_time=migration.start_time,
                end_time=migration.end_time,
                rate=migration.rate,
            )

    return b.resolve()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} demes_graph.yml")
        exit(1)

    g = demes.load(sys.argv[1])
    pc, de, mm = to_msprime(g)

    ddb = msprime.DemographyDebugger(
        population_configurations=pc,
        demographic_events=de,
        migration_matrix=mm,
    )
    ddb.print_history()
