import copy
import decimal
import enum
import fractions
import pathlib
import tempfile
import textwrap

import pytest
import hypothesis as hyp
import numpy as np

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

    b = demes.Builder(
        description="Jacobs et al. (2019) archaic admixture into Papuans",
        doi=[
            "https://doi.org/10.1016/j.cell.2019.02.035",
            "https://doi.org/10.1038/nature18299",
        ],
        time_units="generations",
    )
    b.add_deme("ancestral_hominin", epochs=[dict(end_time=20225, start_size=32671)])
    b.add_deme(
        "archaic",
        ancestors=["ancestral_hominin"],
        epochs=[dict(end_time=15090, start_size=N_archaic)],
    )

    b.add_deme(
        "Den1",
        ancestors=["archaic"],
        epochs=[dict(end_time=12500, start_size=N_DeniAnc)],
    )
    b.add_deme(
        "Den2",
        ancestors=["Den1"],
        epochs=[dict(end_time=9750, start_size=N_DeniAnc)],
    )
    # Altai Denisovan (sampling lineage)
    b.add_deme(
        "DenAltai", ancestors=["Den2"], epochs=[dict(start_size=5083, end_time=0)]
    )
    # Introgressing Denisovan lineages 1 and 2
    b.add_deme(
        "DenI1", ancestors=["Den2"], epochs=[dict(start_size=N_archaic, end_time=0)]
    )
    b.add_deme(
        "DenI2", ancestors=["Den1"], epochs=[dict(start_size=N_archaic, end_time=0)]
    )

    b.add_deme(
        "Nea",
        ancestors=["archaic"],
        epochs=[dict(end_time=3375, start_size=N_archaic)],
    )
    # Altai Neanderthal (sampling lineage)
    b.add_deme("NeaAltai", ancestors=["Nea"], epochs=[dict(start_size=826, end_time=0)])
    # Introgressing Neanderthal lineage
    b.add_deme(
        "NeaI", ancestors=["Nea"], epochs=[dict(end_time=883, start_size=N_archaic)]
    )

    b.add_deme(
        "AMH",
        ancestors=["ancestral_hominin"],
        epochs=[dict(end_time=2218, start_size=41563)],
    )
    b.add_deme("Africa", ancestors=["AMH"], epochs=[dict(start_size=48433, end_time=0)])
    b.add_deme(
        "Ghost1",
        ancestors=["AMH"],
        epochs=[
            # bottleneck
            dict(end_time=2119, start_size=1394),
            dict(end_time=1784, start_size=N_ghost),
        ],
    )
    b.add_deme(
        "Ghost2",
        ancestors=["Ghost1"],
        epochs=[dict(end_time=1758, start_size=N_ghost)],
    )
    b.add_deme(
        "Ghost3", ancestors=["Ghost2"], epochs=[dict(start_size=N_ghost, end_time=0)]
    )
    b.add_deme(
        "Papua",
        ancestors=["Ghost1"],
        # bottleneck
        epochs=[
            dict(end_time=1685, start_size=243),
            dict(end_time=0, start_size=8834),
        ],
    )
    b.add_deme(
        "Eurasia",
        ancestors=["Ghost2"],
        # bottleneck
        epochs=[
            dict(end_time=T_Eu_bottleneck, start_size=2231),
            dict(end_time=1293, start_size=12971),
        ],
    )
    b.add_deme(
        "WestEurasia",
        ancestors=["Eurasia"],
        epochs=[dict(start_size=6962, end_time=0)],
    )
    b.add_deme(
        "EastAsia", ancestors=["Eurasia"], epochs=[dict(start_size=9025, end_time=0)]
    )

    b.add_migration(
        demes=["Africa", "Ghost3"], rate=1.79e-4, start_time=T_Eu_bottleneck
    )
    b.add_migration(demes=["Ghost3", "WestEurasia"], rate=4.42e-4)
    b.add_migration(demes=["WestEurasia", "EastAsia"], rate=3.14e-5)
    b.add_migration(demes=["EastAsia", "Papua"], rate=5.72e-5)
    b.add_migration(
        demes=["Eurasia", "Papua"], rate=5.72e-4, start_time=T_Eu_bottleneck
    )
    b.add_migration(
        demes=["Ghost3", "Eurasia"], rate=4.42e-4, start_time=T_Eu_bottleneck
    )

    b.add_pulse(source="NeaI", dest="EastAsia", proportion=0.002, time=883)
    b.add_pulse(source="NeaI", dest="Papua", proportion=0.002, time=1412)
    b.add_pulse(source="NeaI", dest="Eurasia", proportion=0.011, time=1566)
    b.add_pulse(source="NeaI", dest="Ghost1", proportion=0.024, time=1853)

    m_Den_Papuan = 0.04
    p = 0.55  # S10.i p. 31
    T_Den1_Papuan_mig = 29.8e3 / generation_time
    T_Den2_Papuan_mig = 45.7e3 / generation_time
    b.add_pulse(
        source="DenI1",
        dest="Papua",
        proportion=p * m_Den_Papuan,
        time=T_Den1_Papuan_mig,
    )
    b.add_pulse(
        source="DenI2",
        dest="Papua",
        proportion=(1 - p) * m_Den_Papuan,
        time=T_Den2_Papuan_mig,
    )

    return b.resolve()


class TestLoadAndDump:
    def test_bad_format_param(self):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        ex = next(examples_path.glob("*.yml"))
        with open(ex) as f:
            ex_string = f.read()

        with pytest.raises(ValueError):
            demes.load(ex, format="not a format")
        with pytest.raises(ValueError):
            demes.loads(ex_string, format="not a format")

        g = demes.loads(ex_string)
        for simplified in [True, False]:
            with pytest.raises(ValueError):
                demes.dumps(g, format="not a format", simplified=simplified)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = pathlib.Path(tmpdir) / "never-created"
                with pytest.raises(ValueError):
                    demes.dump(g, tmpfile, format="not a format", simplified=simplified)

    def test_bad_filename_param(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=0)])
        g = b.resolve()

        class F:
            pass

        f_w = F()
        f_w.write = True
        f_r = F()
        f_r.read = None
        for bad_file in [None, -1, object(), f_w, f_r]:
            # There are a variety of exceptions that could be raised here,
            # including AttributeError, ValueError, TypeError, OSError,
            # and probably others. The exact exception is the user's concern,
            # and we just want to check that some obviously wrong files aren't
            # silently accepted.
            with pytest.raises(Exception):
                demes.dump(g, bad_file)
            with pytest.raises(Exception):
                demes.load(bad_file)

    def check_dumps_simple(self, *, format, simplified):
        b1 = demes.Builder(
            description="some very concise descr",
            time_units="years",
            generation_time=42,
        )
        for name, N in zip("ABCD", [100, 200, 300, 400]):
            b1.add_deme(name, epochs=[dict(start_size=N, end_time=0)])
        g1 = b1.resolve()
        string = demes.dumps(g1, format=format, simplified=simplified)
        assert "description" in string
        assert g1.description in string
        assert "time_units" in string
        assert g1.time_units in string
        assert "generation_time" in string
        assert str(g1.generation_time) in string
        assert "demes" in string
        assert "A" in string
        assert "B" in string
        assert "C" in string
        assert "D" in string
        assert "start_size" in string
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

        b2 = copy.deepcopy(b1)
        b2.add_deme("E", epochs=[dict(start_size=100, selfing_rate=0.1)])
        g2 = b2.resolve()
        string = demes.dumps(g2, format=format, simplified=simplified)
        assert "selfing_rate" in string
        assert "0.1" in string
        if simplified:
            assert "cloning_rate" not in string

        b2 = copy.deepcopy(b1)
        b2.add_deme("E", epochs=[dict(start_size=100, cloning_rate=0.1)])
        g2 = b2.resolve()
        string = demes.dumps(g2, format=format, simplified=simplified)
        if simplified:
            assert "selfing_rate" not in string
        assert "cloning_rate" in string
        assert "0.1" in string

    def check_dumps_complex(self, *, format, simplified):
        g = jacobs_papuans()
        string = demes.dumps(g, format=format, simplified=simplified)
        assert "description" in string
        assert g.description in string
        assert "time_units" in string
        assert g.time_units in string
        assert "demes" in string
        for deme in g.demes:
            assert f"{deme.name}" in string
        assert "pulses" in string
        for pulse in g.pulses:
            assert "source" in string
            assert pulse.source in string
            assert "dest" in string
            assert pulse.dest in string
        assert "migrations" in string

    def test_dumps_yaml(self):
        for simplified in [True, False]:
            self.check_dumps_simple(format="yaml", simplified=simplified)
            self.check_dumps_complex(format="yaml", simplified=simplified)

    def test_dumps_json(self):
        for simplified in [True, False]:
            self.check_dumps_simple(format="json", simplified=simplified)
            self.check_dumps_complex(format="json", simplified=simplified)

    def check_dump_against_dumps(self, *, format, simplified):
        g = jacobs_papuans()
        dumps_str = demes.dumps(g, format=format, simplified=simplified)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile1 = pathlib.Path(tmpdir) / "temp1.yml"
            # tmpfile is os.PathLike
            demes.dump(g, tmpfile1, format=format, simplified=simplified)
            with open(tmpfile1) as f:
                yaml_str1 = f.read()
            assert yaml_str1 == dumps_str

            tmpfile2 = pathlib.Path(tmpdir) / "temp2.yml"
            # tmpfile is str
            demes.dump(g, str(tmpfile2), format=format, simplified=simplified)
            with open(tmpfile2) as f:
                yaml_str2 = f.read()
            assert yaml_str2 == dumps_str

    def test_dump_against_dumps(self):
        for simplified in [True, False]:
            self.check_dump_against_dumps(format="yaml", simplified=simplified)
            self.check_dump_against_dumps(format="json", simplified=simplified)

    def test_loads_json_simple(self):
        string = textwrap.dedent(
            """\
            {
                "description": "foo",
                "time_units": "years",
                "generation_time": 123,
                "demes": [
                    {
                        "name": "A",
                        "epochs": [
                        {
                            "start_size": 100,
                            "end_time": 0
                        }
                        ]
                    },
                    {
                        "name": "B",
                        "epochs": [
                        {
                            "start_size": 100,
                            "end_time": 0
                        }
                        ]
                    },
                    {
                        "name": "C",
                        "ancestors": [ "A", "B" ],
                        "proportions": [ 0.1, 0.9 ],
                        "start_time": 500,
                        "epochs": [
                        {
                            "start_size": 100,
                            "end_time": 0
                        }
                        ]
                    }
                ]
            }
            """
        )
        g = demes.loads(string, format="json")
        assert g.description == "foo"
        assert g.time_units == "years"
        assert g.generation_time == 123
        assert [deme.name for deme in g.demes] == ["A", "B", "C"]
        assert g["C"].start_time == 500
        assert g["C"].ancestors == ["A", "B"]
        assert g["C"].proportions == [0.1, 0.9]

    def test_loads_yaml_simple(self):
        string = textwrap.dedent(
            """\
            description: foo
            time_units: years
            generation_time: 123
            demes:
                -   name: A
                    epochs:
                        -   start_size: 100
                            end_time: 0
                -   name: B
                    epochs:
                        -   start_size: 100
                            end_time: 0
                -   name: C
                    start_time: 500
                    epochs:
                        -   start_size: 100
                            end_time: 0
                    ancestors: [A, B]
                    proportions: [0.1, 0.9]
            """
        )
        g = demes.loads(string)
        assert g.description == "foo"
        assert g.time_units == "years"
        assert g.generation_time == 123
        assert [deme.name for deme in g.demes] == ["A", "B", "C"]
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

    def check_dump_and_load_simple(self, *, format, simplified):
        b1 = demes.Builder(
            description="some very concise description",
            time_units="years",
            generation_time=42,
        )
        for name, N in zip("ABCD", [100, 200, 300, 400]):
            b1.add_deme(name, epochs=[dict(start_size=N, end_time=0)])
        g1 = b1.resolve()
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
                # dump and load files
                demes.dump(g1, tmpfile, format=format, simplified=simplified)
                g2 = demes.load(tmpfile, format=format)
                # dump and load via file streams
                with open(tmpfile, "w") as f:
                    demes.dump(g1, f, format=format, simplified=simplified)
                with open(tmpfile) as f:
                    g3 = demes.load(f, format=format)
            assert g1.isclose(g2)
            assert g1.isclose(g3)
            n += 1
        assert n > 1

    def test_examples_load_dump_load_yaml(self):
        for simplified in [True, False]:
            self.check_examples_load_dump_load(format="yaml", simplified=simplified)

    def test_examples_load_dump_load_json(self):
        for simplified in [True, False]:
            self.check_examples_load_dump_load(format="json", simplified=simplified)

    def check_yaml_output_is_pretty(self, g, yamlfile, simplified):
        with open(yamlfile) as f:
            string = f.read()
        # Check for non-human-readable output in the yaml file.
        assert "!!python" not in string
        assert "!!binary" not in string
        assert "!!map" not in string
        assert "!!omap" not in string

        # Check the keys in the yaml are in the same order as our attrs class
        # attributes. I.e. the same order we get from attr.asdict().
        def deep_key_order(a, b):
            assert list(a.keys()) == list(b.keys())
            for k, a_k in a.items():
                if isinstance(a_k, dict):
                    deep_key_order(a_k, b[k])

        data = demes.loads_asdict(string)
        if simplified:
            ref_data = g.asdict_simplified()
        else:
            ref_data = g.asdict()
        deep_key_order(ref_data, data)

    def check_dump_load_roundtrip(self, g):
        with tempfile.TemporaryDirectory() as tmpdir:
            for format in ["yaml", "json"]:
                for simplified in [True, False]:
                    tmpfile = pathlib.Path(tmpdir) / "temp.txt"
                    demes.dump(g, tmpfile, format=format, simplified=simplified)
                    g2 = demes.load(tmpfile, format=format)
                    g.assert_close(g2)

                    if format == "yaml":
                        self.check_yaml_output_is_pretty(g, tmpfile, simplified)

    @hyp.settings(deadline=None, suppress_health_check=[hyp.HealthCheck.too_slow])
    @hyp.given(tests.graphs())
    def test_dump_load(self, g):
        self.check_dump_load_roundtrip(g)

    def test_int_subclass(self):
        # Check that subclasses of int are round-trippable.
        class Ne(enum.IntEnum):
            INITIAL = 1000
            BOTTLENECK = 500
            NOMINAL = 10000
            HUGE = 100000

        b = demes.Builder()
        b.add_deme("ancestral", epochs=[dict(start_size=100)])
        b.add_deme(
            "A",
            start_time=600,
            ancestors=["ancestral"],
            epochs=[
                dict(end_time=500, start_size=Ne.INITIAL),
                dict(end_time=400, start_size=Ne.BOTTLENECK),
                dict(end_time=300, start_size=Ne.NOMINAL),
                dict(end_time=200, start_size=Ne.HUGE),
            ],
        )
        self.check_dump_load_roundtrip(b.resolve())

        N = np.array([Ne.INITIAL, Ne.BOTTLENECK, Ne.NOMINAL, Ne.HUGE], dtype=np.int32)
        T = np.array([600, 500, 400, 300, 200], dtype=np.int64)
        b.add_deme(
            "B",
            start_time=T[0],
            ancestors=["ancestral"],
            epochs=[
                dict(end_time=T[1], start_size=N[0]),
                dict(end_time=T[2], start_size=N[1]),
                dict(end_time=T[3], start_size=N[2]),
                dict(end_time=T[4], start_size=N[3]),
            ],
        )
        self.check_dump_load_roundtrip(b.resolve())

    def test_float_subclass(self):
        # Check that subclasses of float are round-trippable.
        generation_time = np.array([1], dtype=np.float64)
        N = np.array([1000, 500, 10000, 100000], dtype=np.float64)
        T = np.array([600, 500, 400, 300, 200], dtype=np.float32)
        b = demes.Builder(
            description="test", time_units="years", generation_time=generation_time[0]
        )
        b.add_deme("ancestral", epochs=[dict(start_size=100)])
        b.add_deme(
            "A",
            start_time=T[0],
            ancestors=["ancestral"],
            epochs=[
                dict(end_time=T[1], start_size=N[0]),
                dict(end_time=T[2], start_size=N[1]),
                dict(end_time=T[3], start_size=N[2]),
                dict(end_time=T[4], start_size=N[3]),
            ],
        )
        self.check_dump_load_roundtrip(b.resolve())

        b.add_deme("B", epochs=[dict(start_size=N[0], end_time=0)])
        b.add_deme(
            "C",
            ancestors=["A", "B"],
            proportions=[fractions.Fraction(1, 3), fractions.Fraction(2, 3)],
            start_time=T[1],
            epochs=[dict(start_size=N[0], end_time=0)],
        )
        self.check_dump_load_roundtrip(b.resolve())

        b.add_pulse(
            source="A",
            dest="B",
            time=T[1],
            proportion=decimal.Decimal("0.0022"),
        )
        self.check_dump_load_roundtrip(b.resolve())

        b.add_migration(
            source="A",
            dest="B",
            start_time=T[1],
            end_time=T[2],
            rate=decimal.Decimal("0.000012345"),
        )
        self.check_dump_load_roundtrip(b.resolve())
