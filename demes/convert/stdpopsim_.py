import stdpopsim

import demes
from demes.convert import to_msprime, from_msprime


def to_stdpopsim(graph: demes.Graph) -> stdpopsim.DemographicModel:
    """
    Convert a demes graph to a stdpopsim demographic model.

    :param graph: the demes graph to convert.
    :type graph: :class:`demes.Graph`
    :return: A stdpopsim demographic model.
    :rtype demographic_model: :class:`stdpopsim.DemographicModel`
    """
    pc, de, mm = to_msprime(graph)
    return stdpopsim.DemographicModel(
        id="",
        description="Converted from demes.Graph; see long_description.",
        long_description=graph.description,
        citations=[
            stdpopsim.Citation(author=f"Unknown_{j}", year="0000", doi=doi)
            for j, doi in enumerate(graph.doi)
        ],
        generation_time=1,
        populations=[
            stdpopsim.Population(deme.name, deme.description) for deme in graph.demes
        ],
        population_configurations=pc,
        demographic_events=de,
        migration_matrix=mm,
    )


def from_stdpopsim(demographic_model: stdpopsim.DemographicModel) -> demes.Graph:
    """
    Convert a stdpopsim demographic model into a demes graph.

    :param demographic_model: A stdpopsim demographic model.
    :type demographic_model: :class:`stdpopsim.DemographicModel`
    :return: A demes graph.
    :rtype: :class:`demes.Graph`
    """
    g = from_msprime(
        population_configurations=demographic_model.population_configurations,
        demographic_events=demographic_model.demographic_events,
        migration_matrix=demographic_model.migration_matrix,
        pop_names=[pc.id for pc in demographic_model.populations],
    )

    g.description = " ".join(
        s.strip() for s in demographic_model.long_description.splitlines() if s.strip()
    )
    g.doi = [cite.doi for cite in demographic_model.citations]
    return g


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"usage {sys.argv[0]} species demographic_model")
        exit(1)

    species = stdpopsim.get_species(sys.argv[1])
    dm = species.get_demographic_model(sys.argv[2])
    g = from_stdpopsim(dm)

    dm2 = to_stdpopsim(g)
    # dm2.get_demography_debugger().print_history()
    g2 = from_stdpopsim(dm2)
    # print(demes.dumps(g2))

    assert g.isclose(g2)

    g3 = demes.loads(demes.dumps(g))
    assert g.isclose(g3)

    print(demes.dumps(g))
