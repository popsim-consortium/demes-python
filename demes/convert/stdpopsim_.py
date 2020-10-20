import textwrap

import stdpopsim

import demes
from demes.convert import to_msprime, from_msprime


def to_stdpopsim(deme_graph: demes.DemeGraph) -> stdpopsim.DemographicModel:
    """
    Convert a demes graph to a stdpopsim demographic model.

    :param deme_graph: the demes graph to convert.
    :type deme_graph: :class:`demes.DemeGraph`
    :return: A stdpopsim demographic model.
    :rtype demographic_model: :class:`stdpopsim.DemographicModel`
    """
    pc, de, mm = to_msprime(deme_graph)
    return stdpopsim.DemographicModel(
        id="",
        description="Converted from demes.DemeGraph; see long_description.",
        long_description=deme_graph.description,
        citations=[
            stdpopsim.Citation(author="Unknown", year="1234", doi=deme_graph.doi)
        ],
        generation_time=1,
        populations=[
            stdpopsim.Population(deme.id, deme.description) for deme in deme_graph.demes
        ],
        population_configurations=pc,
        demographic_events=de,
        migration_matrix=mm,
    )


def from_stdpopsim(demographic_model: stdpopsim.DemographicModel) -> demes.DemeGraph:
    """
    Convert a stdpopsim demographic model into a demes graph.

    :param demographic_model: A stdpopsim demographic model.
    :type demographic_model: :class:`stdpopsim.DemographicModel`
    :return: A demes graph.
    :rtype: :class:`demes.DemeGraph`
    """
    g = from_msprime(
        population_configurations=demographic_model.population_configurations,
        demographic_events=demographic_model.demographic_events,
        migration_matrix=demographic_model.migration_matrix,
        pop_names=[pc.id for pc in demographic_model.populations],
    )

    g.description = textwrap.dedent(demographic_model.long_description).strip()
    # The doi field is a free-form string, so just dump the string-ified
    # citations in there.
    g.doi = "\n".join([str(cite) for cite in demographic_model.citations])
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

    assert g == g2

    print(demes.dumps(g))
