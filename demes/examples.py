from demes import DemeGraph, Epoch


def zigzag():
    N1 = 14312
    N2 = 1431
    g = DemeGraph(
        description="Sequential exponential increase and decrease of population size.",
        doi="10.1038/ng.3015",
        time_units="years",
        generation_time=30,
    )
    g.deme(
        "ZigZag",
        initial_size=N1,
        epochs=[
            Epoch(start_time=33.33, final_size=N2),
            Epoch(start_time=133.33, final_size=N1),
            Epoch(start_time=533.33, final_size=N2),
            Epoch(start_time=2133.33, final_size=N1),
            Epoch(start_time=8533.33, final_size=N2),
            Epoch(start_time=34133.33, initial_size=N2),
        ],
    )
    return g


def gutenkunst_ooa():
    g = DemeGraph(
        description="Gutenkunst et al. (2009) three-population model.",
        doi="10.1371/journal.pgen.1000695",
        time_units="years",
        generation_time=25,
    )
    g.deme("ancestral", start_time=220e3, initial_size=7300)
    g.deme("AMH", ancestor="ancestral", start_time=140e3, initial_size=12300)
    g.deme("OOA", ancestor="AMH", start_time=21.2e3, initial_size=2100)
    g.deme("YRI", ancestor="AMH", initial_size=12300)
    g.deme("CEU", ancestor="OOA", initial_size=29725, final_size=1000)
    g.deme("CHB", ancestor="OOA", initial_size=54090, final_size=510)
    g.symmetric_migration("YRI", "OOA", rate=25e-5 / g.generation_time)
    g.symmetric_migration("YRI", "CEU", rate=3e-5 / g.generation_time)
    g.symmetric_migration("YRI", "CHB", rate=1.9e-5 / g.generation_time)
    g.symmetric_migration("CEU", "CHB", rate=9.6e-5 / g.generation_time)
    return g


def browning_america():
    g = DemeGraph(
        description="Browning et al. (2011) model of admixture in the Americas.",
        doi="10.1371/journal.pgen.1007385",
        time_units="generations",
        generation_time=25,
        # generation_time_unit="years",
    )
    g.deme("ancestral", start_time=5920, initial_size=7310)
    g.deme("AMH", ancestor="ancestral", start_time=2040, initial_size=14474)
    g.deme("AFR", ancestor="AMH", initial_size=14474)
    g.deme("OOA", ancestor="AMH", start_time=920, initial_size=1861)
    g.deme("EUR", ancestor="OOA", initial_size=34039, final_size=1000)
    g.deme("EAS", ancestor="OOA", initial_size=45852, final_size=510)
    g.subgraph(
        "ADMIX",
        ancestors=["AFR", "EUR", "EAS"],
        proportions=[1 / 6, 1 / 3, 1 / 2],
        end_time=12,
        initial_size=54664,
        final_size=30000,
    )
    g.symmetric_migration("AFR", "OOA", rate=15e-5)
    g.symmetric_migration("AFR", "EUR", rate=2.5e-5)
    g.symmetric_migration("AFR", "EAS", rate=0.78e-5)
    g.symmetric_migration("EUR", "EAS", rate=3.11e-5)
    return g


def jacobs_papuans():
    N_archaic = 13249
    N_DeniAnc = 100
    N_ghost = 8516
    T_Eu_bottleneck = 1659

    g = DemeGraph(
        description="Jacobs et al. (2019) archaic admixture into Papuans",
        doi="10.1016/j.cell.2019.02.035",
        time_units="generations",
        generation_time=29,
    )
    g.deme("ancestral_hominin", start_time=20225, initial_size=32671)
    g.deme(
        "archaic",
        ancestor="ancestral_hominin",
        start_time=15090,
        initial_size=N_archaic,
    )

    g.deme("Den1", ancestor="archaic", start_time=12500, initial_size=N_DeniAnc)
    g.deme("Den2", ancestor="Den1", start_time=9750, initial_size=N_DeniAnc)
    # Altai Denisovan (sampling lineage)
    g.deme("DenAltai", ancestor="Den2", initial_size=5083)
    # Introgressing Denisovan lineages 1 and 2
    g.deme("DenI1", ancestor="Den2", initial_size=N_archaic)
    g.deme("DenI2", ancestor="Den1", initial_size=N_archaic)

    g.deme("Nea", ancestor="archaic", start_time=3375, initial_size=N_archaic)
    # Altai Neanderthal (sampling lineage)
    g.deme("NeaAltai", ancestor="Nea", initial_size=826)
    # Introgressing Neanderthal lineage
    g.deme("NeaI", ancestor="Nea", start_time=883, initial_size=N_archaic)

    g.deme("AMH", ancestor="ancestral_hominin", start_time=2218, initial_size=41563)
    g.deme("Africa", ancestor="AMH", initial_size=48433)
    g.deme(
        "Ghost1",
        ancestor="AMH",
        start_time=1784,
        initial_size=N_ghost,
        epochs=[
            # bottleneck
            Epoch(start_time=2119, initial_size=1394),
            Epoch(start_time=2218, initial_size=N_ghost),
        ],
    )
    g.deme("Ghost2", ancestor="Ghost1", start_time=1758, initial_size=N_ghost)
    g.deme("Ghost3", ancestor="Ghost2", initial_size=N_ghost)
    g.deme(
        "Papua",
        ancestor="Ghost1",
        initial_size=8834,
        # bottleneck
        epochs=[Epoch(start_time=1685, initial_size=243)],
    )
    g.deme(
        "Eurasia",
        ancestor="Ghost2",
        start_time=1293,
        initial_size=12971,
        # bottleneck
        epochs=[Epoch(start_time=T_Eu_bottleneck, initial_size=2231)],
    )
    g.deme("WestEurasia", ancestor="Eurasia", initial_size=6962)
    g.deme("EastAsia", ancestor="Eurasia", initial_size=9025)

    g.symmetric_migration("Africa", "Ghost3", rate=1.79e-4, end_time=T_Eu_bottleneck)
    g.symmetric_migration("Ghost3", "WestEurasia", rate=4.42e-4)
    g.symmetric_migration("WestEurasia", "EastAsia", rate=3.14e-5)
    g.symmetric_migration("EastAsia", "Papua", rate=5.72e-5)
    g.symmetric_migration("Eurasia", "Papua", rate=5.72e-4, end_time=T_Eu_bottleneck)
    g.symmetric_migration("Ghost3", "Eurasia", rate=4.42e-4, end_time=T_Eu_bottleneck)

    g.pulse(source="EastAsia", dest="NeaI", proportion=0.002, time=883)
    g.pulse(source="Papua", dest="NeaI", proportion=0.002, time=1412)
    g.pulse(source="Eurasia", dest="NeaI", proportion=0.011, time=1566)
    g.pulse(source="Ghost1", dest="NeaI", proportion=0.024, time=1853)

    m_Den_Papuan = 0.04
    p = 0.55  # S10.i p. 31
    T_Den1_Papuan_mig = 29.8e3 / g.generation_time
    T_Den2_Papuan_mig = 45.7e3 / g.generation_time
    g.pulse(
        source="Papua",
        dest="DenI1",
        proportion=p * m_Den_Papuan,
        time=T_Den1_Papuan_mig,
    )
    g.pulse(
        source="Papua",
        dest="DenI2",
        proportion=(1 - p) * m_Den_Papuan,
        time=T_Den2_Papuan_mig,
    )

    return g


def IM(num_demes, N, m, time_units="generations", generation_time=1):
    g = DemeGraph(
        description="Isolation with migration.",
        time_units=time_units,
        generation_time=generation_time,
        default_Ne=N,
    )
    for k in range(num_demes):
        g.deme(f"D{k}")
    g.symmetric_migration(*[deme.id for deme in g.demes], rate=m)
    return g
