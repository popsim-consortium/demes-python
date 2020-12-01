import copy
import io
import pathlib
import tempfile
import textwrap

import pytest
import hypothesis as hyp

import demes
import tests


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

    g = demes.Graph(
        description="Jacobs et al. (2019) archaic admixture into Papuans",
        doi=[
            "https://doi.org/10.1016/j.cell.2019.02.035",
            "https://doi.org/10.1038/nature18299",
        ],
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


class TestLoadAndDump:
    def test_bad_format_param(self):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        ex = next(examples_path.glob("*.yml"))
        with pytest.raises(ValueError):
            demes.load(ex, format="not a format")
        g = demes.load(ex)
        for simplified in [True, False]:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = pathlib.Path(tmpdir) / "never-created"
                with pytest.raises(ValueError):
                    demes.dump(g, tmpfile, format="not a format", simplified=simplified)

    def check_dump_simple(self, *, format, simplified):
        g = demes.Graph(
            description="some very concise descr",
            time_units="years",
            generation_time=42,
        )
        for id, N in zip("ABCD", [100, 200, 300, 400]):
            g.deme(id, initial_size=N)
        stream = io.StringIO()
        demes.dump(g, stream, format=format, simplified=simplified)
        string = stream.getvalue()
        assert "description" in string
        assert g.description in string
        assert "time_units" in string
        assert g.time_units in string
        assert "generation_time" in string
        assert str(g.generation_time) in string
        assert "demes" in string
        assert "A" in string
        assert "B" in string
        assert "C" in string
        assert "D" in string
        assert "initial_size" in string
        assert str(100) in string
        assert str(200) in string
        assert str(300) in string
        assert str(400) in string

        if simplified:
            assert "doi" not in string
            assert "migrations" not in string
            assert "asymmetric" not in string
            assert "symmetric" not in string
            assert "pulses" not in string
            assert "selfing_rate" not in string
            assert "cloning_rate" not in string

        g1 = copy.deepcopy(g)
        g1.deme("E", initial_size=100, selfing_rate=0.1)
        stream = io.StringIO()
        demes.dump(g1, stream, format=format, simplified=simplified)
        string = stream.getvalue()
        assert "selfing_rate" in string
        assert "0.1" in string
        if simplified:
            assert "cloning_rate" not in string

        g1 = copy.deepcopy(g)
        g1.deme("E", initial_size=100, cloning_rate=0.1)
        stream = io.StringIO()
        demes.dump(g1, stream, format=format, simplified=simplified)
        string = stream.getvalue()
        if simplified:
            assert "selfing_rate" not in string
        assert "cloning_rate" in string
        assert "0.1" in string

    def check_dump_complex(self, *, format, simplified):
        g = jacobs_papuans()
        stream = io.StringIO()
        demes.dump(g, stream, format=format, simplified=simplified)
        string = stream.getvalue()
        assert "description" in string
        assert g.description in string
        assert "time_units" in string
        assert g.time_units in string
        assert "demes" in string
        for deme in g.demes:
            assert f"{deme.id}" in string
        assert "pulses" in string
        for pulse in g.pulses:
            assert "source" in string
            assert pulse.source in string
            assert "dest" in string
            assert pulse.dest in string
        assert "migrations" in string
        if simplified:
            assert "asymmetric" not in string
            assert "symmetric" in string
        else:
            assert "asymmetric" in string

    def test_dump_yaml(self):
        for simplified in [True, False]:
            self.check_dump_simple(format="yaml", simplified=simplified)
            self.check_dump_complex(format="yaml", simplified=simplified)

    def test_dump_json(self):
        for simplified in [True, False]:
            self.check_dump_simple(format="json", simplified=simplified)
            self.check_dump_complex(format="json", simplified=simplified)

    def check_dump_str_path_fileobj(self, *, format, simplified):
        g = jacobs_papuans()
        stream = io.StringIO()
        demes.dump(g, stream, format=format, simplified=simplified)
        dump_str = stream.getvalue()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile1 = pathlib.Path(tmpdir) / "temp1.yml"
            # tmpfile is os.PathLike
            demes.dump(g, tmpfile1, format=format, simplified=simplified)
            with open(tmpfile1) as f:
                yaml_str1 = f.read()
            assert yaml_str1 == dump_str

            tmpfile2 = pathlib.Path(tmpdir) / "temp2.yml"
            # tmpfile is str
            demes.dump(g, str(tmpfile2), format=format, simplified=simplified)
            with open(tmpfile2) as f:
                yaml_str2 = f.read()
            assert yaml_str2 == dump_str

            tmpfile3 = pathlib.Path(tmpdir) / "temp3.yml"
            # tmpfile is opened, and we dump to file-like object
            with open(tmpfile3, "w") as f:
                demes.dump(g, f, format=format, simplified=simplified)
            with open(tmpfile3) as f:
                yaml_str3 = f.read()
            assert yaml_str3 == dump_str

    def test_dump_str_path_fileobj(self):
        for simplified in [True, False]:
            self.check_dump_str_path_fileobj(format="yaml", simplified=simplified)
            self.check_dump_str_path_fileobj(format="json", simplified=simplified)

    def test_load_json_simple(self):
        string = textwrap.dedent(
            """\
            {
                "description": "foo",
                "time_units": "years",
                "generation_time": 123,
                "demes": [
                    {
                        "id": "A",
                        "initial_size": 100
                    },
                    {
                        "id": "B",
                        "initial_size": 100
                    },
                    {
                        "id": "C",
                        "ancestors": [ "A", "B" ],
                        "proportions": [ 0.1, 0.9 ],
                        "start_time": 500,
                        "initial_size": 100
                    }
                ]
            }
            """
        )
        stream = io.StringIO(string)
        g = demes.load(stream, format="json")
        assert g.description == "foo"
        assert g.time_units == "years"
        assert g.generation_time == 123
        assert [deme.id for deme in g.demes] == ["A", "B", "C"]
        assert g["C"].start_time == 500
        assert g["C"].ancestors == ["A", "B"]
        assert g["C"].proportions == [0.1, 0.9]

    def test_load_yaml_simple(self):
        string = textwrap.dedent(
            """\
            description: foo
            time_units: years
            generation_time: 123
            demes:
                -   id: A
                    initial_size: 100
                -   id: B
                    initial_size: 100
                -   id: C
                    initial_size: 100
                    start_time: 500
                    ancestors: [A, B]
                    proportions: [0.1, 0.9]
            """
        )
        stream = io.StringIO(string)
        g = demes.load(stream)
        assert g.description == "foo"
        assert g.time_units == "years"
        assert g.generation_time == 123
        assert [deme.id for deme in g.demes] == ["A", "B", "C"]
        assert g["C"].start_time == 500
        assert g["C"].ancestors == ["A", "B"]
        assert g["C"].proportions == [0.1, 0.9]

    def test_load_examples(self):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        n = 0
        for yaml_file in examples_path.glob("*.yml"):
            n += 1
            g = demes.load(yaml_file)
            assert g.description is not None
            assert len(g.description) > 0
            assert g.time_units is not None
            assert len(g.time_units) > 0
            assert len(g.demes) > 0
        assert n > 1

    def check_dump_and_load_simple(self, *, format, simplified):
        g1 = demes.Graph(
            description="some very concise description",
            time_units="years",
            generation_time=42,
        )
        for id, N in zip("ABCD", [100, 200, 300, 400]):
            g1.deme(id, initial_size=N)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "temp.txt"
            demes.dump(g1, tmpfile, format=format, simplified=simplified)
            g2 = demes.load(tmpfile, format=format)
        assert g1.isclose(g2)

    def check_dump_and_load_complex(self, *, format, simplified):
        g1 = jacobs_papuans()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "temp.txt"
            demes.dump(g1, tmpfile, format=format, simplified=simplified)
            g2 = demes.load(tmpfile, format=format)
        assert g1.isclose(g2)

    def test_dump_and_load_yaml(self):
        for simplified in [True, False]:
            self.check_dump_and_load_simple(format="yaml", simplified=simplified)
            self.check_dump_and_load_complex(format="yaml", simplified=simplified)

    def test_dump_and_load_json(self):
        for simplified in [True, False]:
            self.check_dump_and_load_simple(format="json", simplified=simplified)
            self.check_dump_and_load_complex(format="json", simplified=simplified)

    def check_examples_load_dump_load(self, *, format, simplified):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        n = 0
        for yaml_file in examples_path.glob("*.yml"):
            g1 = demes.load(yaml_file, format="yaml")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = pathlib.Path(tmpdir) / "temp.yml"
                demes.dump(g1, tmpfile, format=format, simplified=simplified)
                g2 = demes.load(tmpfile, format=format)
            assert g1.isclose(g2)
            n += 1
        assert n > 1

    def test_examples_load_dump_load_yaml(self):
        for simplified in [True, False]:
            self.check_examples_load_dump_load(format="yaml", simplified=simplified)

    def test_examples_load_dump_load_json(self):
        for simplified in [True, False]:
            self.check_examples_load_dump_load(format="json", simplified=simplified)

    @hyp.settings(deadline=None, suppress_health_check=[hyp.HealthCheck.too_slow])
    @hyp.given(tests.graphs())
    def test_dump_load(self, g):
        with tempfile.TemporaryDirectory() as tmpdir:
            for format in ["yaml", "json"]:
                for simplified in [True, False]:
                    tmpfile = pathlib.Path(tmpdir) / "temp.txt"
                    demes.dump(g, tmpfile, format=format, simplified=simplified)
                    g2 = demes.load(tmpfile, format=format)
                    g.assert_close(g2)
