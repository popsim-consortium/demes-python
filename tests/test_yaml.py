import copy
import pathlib
import tempfile
import textwrap

import demes


def jacobs_papuans():
    """
    XXX: This model is for testing only and has not been vetted for accuracy!
         Use examples/jacobs_papuans.yml, or the PapuansOutOfAfrica_10J19 model
         from stdpopsim instead.
    """
    generation_time = 29
    N_archaic = 13249
    N_DeniAnc = 100
    N_ghost = 8516
    T_Eu_bottleneck = 1659

    g = demes.DemeGraph(
        description="Jacobs et al. (2019) archaic admixture into Papuans",
        doi="10.1016/j.cell.2019.02.035",
        time_units="generations",
    )
    g.deme("ancestral_hominin", end_time=20225, initial_size=32671)
    g.deme(
        "archaic",
        ancestors=["ancestral_hominin"],
        end_time=15090,
        initial_size=N_archaic,
    )

    g.deme("Den1", ancestors=["archaic"], end_time=12500, initial_size=N_DeniAnc)
    g.deme("Den2", ancestors=["Den1"], end_time=9750, initial_size=N_DeniAnc)
    # Altai Denisovan (sampling lineage)
    g.deme("DenAltai", ancestors=["Den2"], initial_size=5083)
    # Introgressing Denisovan lineages 1 and 2
    g.deme("DenI1", ancestors=["Den2"], initial_size=N_archaic)
    g.deme("DenI2", ancestors=["Den1"], initial_size=N_archaic)

    g.deme("Nea", ancestors=["archaic"], end_time=3375, initial_size=N_archaic)
    # Altai Neanderthal (sampling lineage)
    g.deme("NeaAltai", ancestors=["Nea"], initial_size=826)
    # Introgressing Neanderthal lineage
    g.deme("NeaI", ancestors=["Nea"], end_time=883, initial_size=N_archaic)

    g.deme("AMH", ancestors=["ancestral_hominin"], end_time=2218, initial_size=41563)
    g.deme("Africa", ancestors=["AMH"], initial_size=48433)
    g.deme(
        "Ghost1",
        ancestors=["AMH"],
        initial_size=N_ghost,
        epochs=[
            # bottleneck
            demes.Epoch(end_time=2119, initial_size=1394),
            demes.Epoch(end_time=1784, initial_size=N_ghost),
        ],
    )
    g.deme("Ghost2", ancestors=["Ghost1"], end_time=1758, initial_size=N_ghost)
    g.deme("Ghost3", ancestors=["Ghost2"], initial_size=N_ghost)
    g.deme(
        "Papua",
        ancestors=["Ghost1"],
        # bottleneck
        epochs=[
            demes.Epoch(end_time=1685, initial_size=243),
            demes.Epoch(end_time=0, initial_size=8834),
        ],
    )
    g.deme(
        "Eurasia",
        ancestors=["Ghost2"],
        # bottleneck
        epochs=[
            demes.Epoch(end_time=T_Eu_bottleneck, initial_size=2231),
            demes.Epoch(end_time=1293, initial_size=12971),
        ],
    )
    g.deme("WestEurasia", ancestors=["Eurasia"], initial_size=6962)
    g.deme("EastAsia", ancestors=["Eurasia"], initial_size=9025)

    g.symmetric_migration(
        demes=["Africa", "Ghost3"], rate=1.79e-4, start_time=T_Eu_bottleneck
    )
    g.symmetric_migration(demes=["Ghost3", "WestEurasia"], rate=4.42e-4)
    g.symmetric_migration(demes=["WestEurasia", "EastAsia"], rate=3.14e-5)
    g.symmetric_migration(demes=["EastAsia", "Papua"], rate=5.72e-5)
    g.symmetric_migration(
        demes=["Eurasia", "Papua"], rate=5.72e-4, start_time=T_Eu_bottleneck
    )
    g.symmetric_migration(
        demes=["Ghost3", "Eurasia"], rate=4.42e-4, start_time=T_Eu_bottleneck
    )

    g.pulse(source="NeaI", dest="EastAsia", proportion=0.002, time=883)
    g.pulse(source="NeaI", dest="Papua", proportion=0.002, time=1412)
    g.pulse(source="NeaI", dest="Eurasia", proportion=0.011, time=1566)
    g.pulse(source="NeaI", dest="Ghost1", proportion=0.024, time=1853)

    m_Den_Papuan = 0.04
    p = 0.55  # S10.i p. 31
    T_Den1_Papuan_mig = 29.8e3 / generation_time
    T_Den2_Papuan_mig = 45.7e3 / generation_time
    g.pulse(
        source="DenI1",
        dest="Papua",
        proportion=p * m_Den_Papuan,
        time=T_Den1_Papuan_mig,
    )
    g.pulse(
        source="DenI2",
        dest="Papua",
        proportion=(1 - p) * m_Den_Papuan,
        time=T_Den2_Papuan_mig,
    )

    return g


class TestYAML:
    def test_dumps_simple(self):
        g = demes.DemeGraph(
            description="some very concise description",
            time_units="years",
            generation_time=42,
        )
        for id, N in zip("ABCD", [100, 200, 300, 400]):
            g.deme(id, initial_size=N)
        yaml_str = demes.dumps(g)
        assert f"description: {g.description}" in yaml_str
        assert f"time_units: {g.time_units}" in yaml_str
        assert f"generation_time: {g.generation_time}" in yaml_str
        assert "demes:" in yaml_str
        assert "A:" in yaml_str
        assert "B:" in yaml_str
        assert "C:" in yaml_str
        assert "D:" in yaml_str
        assert "initial_size: 100" in yaml_str
        assert "initial_size: 200" in yaml_str
        assert "initial_size: 300" in yaml_str
        assert "initial_size: 400" in yaml_str

        assert "doi" not in yaml_str
        assert "migrations" not in yaml_str
        assert "asymmetric" not in yaml_str
        assert "symmetric" not in yaml_str
        assert "pulses" not in yaml_str
        assert "selfing_rate" not in yaml_str
        assert "cloning_rate" not in yaml_str

        g1 = copy.deepcopy(g)
        g1.deme("E", initial_size=100, selfing_rate=0.1)
        yaml_str = demes.dumps(g1)
        assert "selfing_rate: 0.1" in yaml_str
        assert "cloning_rate" not in yaml_str

        g1 = copy.deepcopy(g)
        g1.deme("E", initial_size=100, cloning_rate=0.1)
        yaml_str = demes.dumps(g1)
        assert "selfing_rate" not in yaml_str
        assert "cloning_rate: 0.1" in yaml_str

    def test_dumps_complex(self):
        g = jacobs_papuans()
        yaml_str = demes.dumps(g)
        assert f"description: {g.description}" in yaml_str
        assert f"time_units: {g.time_units}" in yaml_str
        assert f"doi: {g.doi}" in yaml_str
        assert "generation_time" not in yaml_str
        assert "demes:" in yaml_str
        for deme in g.demes:
            assert f"{deme.id}:" in yaml_str
        assert "pulses:" in yaml_str
        for pulse in g.pulses:
            assert f"source: {pulse.source}" in yaml_str
            assert f"dest: {pulse.dest}" in yaml_str
        assert "migrations:" in yaml_str
        assert "asymmetric" not in yaml_str
        assert "symmetric:" in yaml_str

    def test_dump_against_dumps(self):
        g = jacobs_papuans()
        dumps_str = demes.dumps(g)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile1 = pathlib.Path(tmpdir) / "temp1.yml"
            # tmpfile is os.PathLike
            demes.dump(g, tmpfile1)
            with open(tmpfile1) as f:
                yaml_str1 = f.read()
            assert yaml_str1 == dumps_str

            tmpfile2 = pathlib.Path(tmpdir) / "temp2.yml"
            # tmpfile is str
            demes.dump(g, str(tmpfile2))
            with open(tmpfile2) as f:
                yaml_str2 = f.read()
            assert yaml_str2 == dumps_str

    def test_loads_simple(self):
        yaml_str = textwrap.dedent(
            """\
            description: foo
            time_units: years
            generation_time: 123
            demes:
                A:
                    initial_size: 100
                B:
                    initial_size: 100
                C:
                    initial_size: 100
                    start_time: 500
                    ancestors: A, B
                    proportions: 0.1, 0.9
            """
        )
        g = demes.loads(yaml_str)
        assert g.description == "foo"
        assert g.time_units == "years"
        assert g.generation_time == 123
        assert [deme.id for deme in g.demes] == ["A", "B", "C"]
        assert g["C"].start_time == 500
        assert g["C"].ancestors == ["A", "B"]
        assert g["C"].proportions == [0.1, 0.9]

    def test_loads_examples(self):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        n = 0
        for yaml_file in examples_path.glob("*.yml"):
            n += 1
            with open(yaml_file) as f:
                yaml_str = f.read()
            g = demes.loads(yaml_str)
            assert g.description is not None
            assert len(g.description) > 0
            assert g.time_units is not None
            assert len(g.time_units) > 0
            assert len(g.demes) > 0
        assert n > 1

    def test_dump_and_load_simple(self):
        g1 = demes.DemeGraph(
            description="some very concise description",
            time_units="years",
            generation_time=42,
        )
        for id, N in zip("ABCD", [100, 200, 300, 400]):
            g1.deme(id, initial_size=N)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "temp.yml"
            demes.dump(g1, tmpfile)
            g2 = demes.load(tmpfile)
        assert g1.isclose(g2)

    def test_dump_and_load_complex(self):
        g1 = jacobs_papuans()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "temp.yml"
            demes.dump(g1, tmpfile)
            g2 = demes.load(tmpfile)
        assert g1.isclose(g2)

    def test_examples_load_dump_load(self):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        n = 0
        for yaml_file in examples_path.glob("*.yml"):
            g1 = demes.load(yaml_file)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = pathlib.Path(tmpdir) / "temp.yml"
                demes.dump(g1, tmpfile)
                g2 = demes.load(tmpfile)
            assert g1.isclose(g2)
            n += 1
        assert n > 1
