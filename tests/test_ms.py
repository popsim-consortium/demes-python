import collections
import math
import pathlib
import pytest
import tempfile

import demes

cwd = pathlib.Path(__file__).parent.resolve()
example_dir = cwd / ".." / "examples"


class TestFromMs:
    def N0(self, *, theta, mu, length):
        return theta / (4 * mu * length)

    def test_ignored_options_have_no_effect(self):
        def check(command, N0=1):
            # with pytest.warns(UserWarning, match="Ignoring unknown args"):
            graph = demes.from_ms(command, N0=N0)
            assert len(graph.pulses) == 0
            assert len(graph.migrations) == 0
            assert len(graph.demes) == 1
            assert len(graph["deme1"].epochs) == 1
            epoch = graph["deme1"].epochs[0]
            assert math.isclose(epoch.start_size, N0)
            assert math.isclose(epoch.end_size, N0)

        for theta in (1, 100.0, 5e6):
            cmd = f"ms 2 1 -t {theta}"
            check(cmd)

        for seeds in ([1, 2, 3], [123, 321, 678]):
            cmd = f"ms 2 1 -t 1.0 -seeds {seeds[0]} {seeds[1]} {seeds[2]}"
            check(cmd)

        for segsites in (0, 1, 1000):
            cmd = f"ms 2 1 -s {segsites}"
            check(cmd)

        for flag in ("-T", "-L"):
            cmd = f"ms 2 1 {flag}"
            check(cmd)

        for precision in (2, 4, 10):
            cmd = f"ms 2 1 -t 1.0 -p {precision}"
            check(cmd)

        for rho, nsites in ([0, 1000], [2.0, 1000], [50.0, 250e6]):
            cmd = f"ms 2 1 -t 1.0 -r {rho} {nsites}"
            check(cmd)

            for f, lambda_ in ([0, 50], [0.5, 200], [1e-5, 3500]):
                cmd = f"ms 2 1 -t 1.0 -r {rho} {nsites} -c {f} {lambda_}"
                check(cmd)

    def test_empty_command(self):
        # One-deme constant-size model.
        graph = demes.from_ms("", N0=1)
        assert len(graph.pulses) == 0
        assert len(graph.migrations) == 0
        assert len(graph.demes) == 1
        assert len(graph["deme1"].epochs) == 1
        epoch = graph["deme1"].epochs[0]
        assert epoch.start_size == epoch.end_size

    def test_structure(self):
        # -I npop n1 n2 ... [4*N0*m]
        def check(command, num_demes: int, migration: bool, N0=1):
            graph = demes.from_ms(command, N0=N0)
            assert len(graph.pulses) == 0
            if migration:
                assert len(graph.migrations) == num_demes * (num_demes - 1)
            else:
                assert len(graph.migrations) == 0
            assert len(graph.demes) == num_demes
            return graph

        N0 = 100
        for npop in (2, 3, 100):
            sample_config = " 0" * (npop - 1)
            cmd = f"ms 2 1 -I {npop} 2 {sample_config}"
            check(cmd, npop, migration=False, N0=N0)
            for M in (1e-5, 1.0, 400):
                assert M <= 4 * N0, "mutation rate too high, use a larger N0"
                cmd = f"ms 2 1 -I {npop} 2 {sample_config} {M}"
                graph = check(cmd, npop, migration=True, N0=N0)
                # The migration rate used with -I is a bit different to
                # the other migration options. The rate provided corresponds
                # to the sum total of all migrations entering any given deme.
                ingress_rate = collections.defaultdict(lambda: 0)
                rate = M / (4 * N0)
                for migration in graph.migrations:
                    assert math.isclose(migration.rate, rate / (npop - 1))
                    ingress_rate[migration.dest] += migration.rate
                for deme, rate_in in ingress_rate.items():
                    assert math.isclose(rate_in, rate), (
                        f"sum of migrations entering {deme} is {rate_in}, "
                        f"expected {rate}"
                    )

        for bad_npop in (0, -1, 1.1, "x", math.inf):
            cmd = f"ms 2 1 -I {bad_npop} 2 2"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_M in (-1, "x", math.inf):
            cmd = f"ms 2 1 -I 2 1 1 {bad_M}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_sample_configuration in ("50 0 0 2", "2 0 0 0 0 0 2"):
            cmd = "ms 2 1 -t 1.0 -I {bad_sample_configuration}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

    def test_growth_rates(self):
        # -G α
        # -eG t α
        # -g i α
        # -eg t i α
        def check(command, num_demes=1, N0=1):
            graph = demes.from_ms(command, N0=N0)
            assert len(graph.pulses) == 0
            assert len(graph.migrations) == 0
            assert len(graph.demes) == num_demes
            return graph

        graph = check("ms 2 1 -t 1.0 -G 0")
        assert len(graph["deme1"].epochs) == 1
        epoch = graph["deme1"].epochs[0]
        assert epoch.start_size == epoch.end_size == 1

        # To convert model that includes growth rates, there must be two or
        # more epochs, otherwise the growth rate applies up to time=inf,
        # and we can't calculate a start_size for the deme.
        for alpha in (-0.01, 1e-5, 0.01, 5):
            cmd = f"ms 2 1 -t 1.0 -G {alpha}"
            with pytest.raises(ValueError, match="growth rate"):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -eG 0 {alpha}"
            with pytest.raises(ValueError, match="growth rate"):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 2 1 1 -eg 0 2 {alpha}"
            with pytest.raises(ValueError, match="growth rate"):
                demes.from_ms(cmd, N0=1)

        def check_2epoch(cmd, T, N1, num_demes=1, name="deme1", N0=1):
            """
            Check the growth rate was applied in the most recent epoch.
            """
            graph = check(cmd, num_demes=num_demes, N0=N0)
            assert len(graph[name].epochs) == 2
            e1, e2 = graph[name].epochs
            assert e1.start_size == e1.end_size == e2.start_size
            assert math.isclose(e2.start_size, N1)
            assert e2.end_size == N0
            for deme in graph.demes:
                if deme.name == name:
                    continue
                assert len(deme.epochs) == 1
                e1 = deme.epochs[0]
                assert e1.start_size == e1.end_size == N0

        T1 = 10
        T2 = 20
        N0 = 100

        for alpha in (-0.01, 1e-5, 0.01, 5):
            N1 = N0 * math.exp(-alpha * T1)
            cmd = f"ms 2 1 -t 1.0 -G {alpha} -eG {T1} 0"
            check_2epoch(cmd, T1, N1, N0=N0)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -g 1 {alpha} -eG {T1} 0"
            check_2epoch(cmd, T1, N1, N0=N0)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -g 1 {alpha} -eg {T1} 1 0"
            check_2epoch(cmd, T1, N1, N0=N0)

            # Change alpha in deme2
            cmd = f"ms 2 1 -t 1.0 -I 2 1 1 -g 2 {alpha} -eG {T1} 0"
            check_2epoch(cmd, T1, N1, num_demes=2, name="deme2", N0=N0)
            cmd = f"ms 2 1 -t 1.0 -I 2 1 1 -g 2 {alpha} -eg {T1} 2 0"
            check_2epoch(cmd, T1, N1, num_demes=2, name="deme2", N0=N0)

        def check_3epoch(cmd, T1, T2, N1, num_demes=1, name="deme1", N0=1):
            """
            Check the growth rate was applied in the middle epoch.
            """
            graph = check(cmd, num_demes=num_demes, N0=N0)
            assert len(graph[name].epochs) == 3
            e1, e2, e3 = graph[name].epochs
            assert e1.start_size == e1.end_size == e2.start_size
            assert math.isclose(e2.start_size, N1)
            assert e2.end_size == N0
            assert e3.start_size == e3.end_size == N0
            for deme in graph.demes:
                if deme.name == name:
                    continue
                assert len(deme.epochs) == 1
                e1 = deme.epochs[0]
                assert e1.start_size == e1.end_size == N0

        for alpha in (-0.01, 1e-5, 0.01, 5):
            N1 = N0 * math.exp(-alpha * (T2 - T1))
            cmd = f"ms 2 1 -t 1.0 -eG {T1} {alpha} -eG {T2} 0"
            check_3epoch(cmd, T1, T2, N1, N0=N0)
            cmd = f"ms 2 1 -t 1.0 -eg {T1} 1 {alpha} -eG {T2} 0"
            check_3epoch(cmd, T1, T2, N1, N0=N0)
            cmd = f"ms 2 1 -t 1.0 -eG {T1} {alpha} -eg {T2} 1 0"
            check_3epoch(cmd, T1, T2, N1, N0=N0)
            cmd = f"ms 2 1 -t 1.0 -eg {T1} 1 {alpha} -eg {T2} 1 0"
            check_3epoch(cmd, T1, T2, N1, N0=N0)

            # Change alpha in deme2
            cmd = f"ms 2 1 -t 1.0 -I 2 1 1 -eg {T1} 2 {alpha} -eg {T2} 2 0"
            check_3epoch(cmd, T1, T2, N1, num_demes=2, name="deme2", N0=N0)
            cmd = f"ms 2 1 -t 1.0 -I 2 1 1 -eg {T1} 2 {alpha} -eG {T2} 0"
            check_3epoch(cmd, T1, T2, N1, num_demes=2, name="deme2", N0=N0)

        for bad_alpha in ("x", math.inf):
            cmd = f"ms 2 1 -t 1.0 -G {bad_alpha}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -G {bad_alpha} -eG 10 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -g 1 {bad_alpha} -eG 10 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -eG 10 {bad_alpha} -eG 20 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -eg 10 1 {bad_alpha} -eG 20 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_time in ("x", math.inf, -10, -0.1):
            cmd = f"ms 2 1 -t 1.0 -eG {bad_time} 0.01 -eG 10 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -eg {bad_time} 1 0.01 -eG 20 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_popid in ("x", math.inf, 0.5, 1000):
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -eg 0 {bad_popid} 0.01 -eG 20 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -g {bad_popid} 0.01 -eG 10 0"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

    def test_size_changes(self):
        # -n, -en, -eN
        # -eN t x
        # -n i x
        # -en t i x
        def check(command, sizes, N0=1):
            graph = demes.from_ms(command, N0=N0)
            assert len(graph.pulses) == 0
            assert len(graph.migrations) == 0
            assert len(graph.demes) == len(sizes)
            for deme, size in zip(graph.demes, sizes):
                assert len(deme.epochs) == 1
                epoch = deme.epochs[0]
                assert math.isclose(epoch.start_size, size)
                assert math.isclose(epoch.end_size, size)

        for size in (0.1, 2, 10.0, 100):
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -n 1 {size}"
            check(cmd, sizes=[size])
            cmd = f"ms 2 1 -t 1.0 -I 3 2 0 0 -n 2 {size}"
            check(cmd, sizes=[1, size, 1])
            cmd = f"ms 2 1 -t 1.0 -I 3 2 0 0 -n 2 {size} -n 3 {2 * size}"
            check(cmd, sizes=[1, size, 2 * size])
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -en 0 1 {size}"
            check(cmd, sizes=[size])
            cmd = f"ms 2 1 -t 1.0 -I 3 2 0 0 -eN 0 {size}"
            check(cmd, sizes=[size, size, size])

        def check_2epoch(command, T, sizes, N0=1):
            graph = demes.from_ms(command, N0=N0)
            assert len(graph.pulses) == 0
            assert len(graph.migrations) == 0
            assert len(graph.demes) == len(sizes)
            for deme, (N1, N2) in zip(graph.demes, sizes):
                assert len(deme.epochs) == 2
                e1, e2 = deme.epochs
                assert math.isclose(e1.start_size, N1)
                assert math.isclose(e1.end_size, N1)
                assert math.isclose(e1.end_time, T * 4 * N0)
                assert math.isclose(e2.start_size, N2)
                assert math.isclose(e2.end_size, N2)
                assert e2.end_time == 0

        # two epoch
        T = 10
        for size in (0.1, 2, 10.0, 100):
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -en {T} 1 {size}"
            check_2epoch(cmd, T, sizes=[(size, 1)])
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -en 0 1 {size} -en {T} 1 1"
            check_2epoch(cmd, T, sizes=[(1, size)])
            cmd = f"ms 2 1 -t 1.0 -I 3 2 0 0 -eN {T} {size}"
            check_2epoch(cmd, T, sizes=[(size, 1), (size, 1), (size, 1)])
            cmd = f"ms 2 1 -t 1.0 -I 3 2 0 0 -eN 0 {size} -eN {T} 1"
            check_2epoch(cmd, T, sizes=[(1, size), (1, size), (1, size)])

        for bad_size in (-1, "x", math.inf):
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -n 1 {bad_size}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 3 2 0 0 -n 2 {bad_size}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)
            cmd = f"ms 2 1 -t 1.0 -I 1 2 -en 10 1 {bad_size}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

    def test_migration(self):
        # -eM t x
        # -m i j rate
        # -em t i j rate
        # -ma M11 M12 M12 ... M21 ...
        # -ema t npop M11 M12 M12 ... M21 ...
        N0 = 100
        # rate=0 is permissible, but shouldn't result in migration
        for cmd in (
            "-I 2 0 2 -m 1 2 0",
            "-I 2 0 2 -em 0 1 2 0",
            "-I 2 0 2 -em 10 1 2 0",
            "-I 2 0 2 -ma 0 0 0 0",
            "-I 2 0 2 -ma x 0 0 x",
            "-I 2 0 2 -ema 0 2 0 0 0 0",
            "-I 2 0 2 -ema 0 2 x 0 0 x",
            "-I 2 0 2 -ema 10 2 0 0 0 0",
            "-I 2 0 2 -ema 10 2 x 0 0 x",
        ):
            graph = demes.from_ms(cmd, N0=N0)
            assert len(graph.migrations) == 0

        for rate in (0.1, 1e-5):
            # one migration
            for cmd in (
                f"-I 2 0 2 -m 1 2 {rate}",
                f"-I 2 0 2 -em 0 1 2 {rate}",
                f"-I 2 0 2 -em 10 1 2 {rate}",
                f"-I 2 0 2 -ma 0 {rate} 0 0",
                f"-I 2 0 2 -ma x {rate} 0 x",
                f"-I 2 0 2 -ema 0 2 0 {rate} 0 0",
                f"-I 2 0 2 -ema 0 2 x {rate} 0 x",
                f"-I 2 0 2 -ema 10 2 0 {rate} 0 0",
                f"-I 2 0 2 -ema 10 2 x {rate} 0 x",
            ):
                graph = demes.from_ms(cmd, N0=N0)
                assert len(graph.migrations) == 1
                assert math.isclose(graph.migrations[0].rate, rate / (4 * N0))

            # two migrations
            for cmd in (
                f"-I 2 0 2 -m 1 2 {rate} -m 2 1 {rate}",
                f"-I 2 0 2 -em 0 1 2 {rate} -em 0 2 1 {rate}",
                f"-I 2 0 2 -em 10 1 2 {rate} -em 10 2 1 {rate}",
                f"-I 2 0 2 -ma 0 {rate} {rate} 0",
                f"-I 2 0 2 -ma x {rate} {rate} x",
                f"-I 2 0 2 -ema 0 2 0 {rate} {rate} 0",
                f"-I 2 0 2 -ema 0 2 x {rate} {rate} x",
                f"-I 2 0 2 -ema 10 2 0 {rate} {rate} 0",
                f"-I 2 0 2 -ema 10 2 x {rate} {rate} x",
            ):
                graph = demes.from_ms(cmd, N0=N0)
                assert len(graph.migrations) == 2
                for migration in graph.migrations:
                    assert math.isclose(migration.rate, rate / (4 * N0))

            # two migrations at distinct times
            T1 = 10
            T2 = 20
            for on1 in (
                # turn migration on at t=0
                f"-m 1 2 {rate}",
                f"-ma 0 {rate} 0 0",
                f"-em 0 1 2 {rate}",
                f"-ema 0 2 0 {rate} 0 0",
            ):
                for off in (
                    # turn migration off at t=T1
                    f"-em {T1} 1 2 0",
                    f"-ema {T1} 2 0 0 0 0",
                ):
                    for on2 in (
                        # turn migration on again, with twice the rate, at t=T2
                        f"-em {T2} 1 2 {rate*2}",
                        f"-ema {T2} 2 0 {rate*2} 0 0",
                    ):

                        cmd = f"-I 2 0 2 {on1} {off} {on2}"
                        graph = demes.from_ms(cmd, N0=N0)
                        assert len(graph.migrations) == 2
                        # Check the migration details, but avoid depending on the
                        # ordering of the migrations list.
                        m1, m2 = sorted(graph.migrations, key=lambda m: m.end_time)
                        assert m1.end_time == 0
                        assert math.isclose(T1 * 4 * N0, m1.start_time)
                        assert math.isclose(T2 * 4 * N0, m2.end_time)
                        assert math.isinf(m2.start_time)
                        assert math.isclose(m1.rate, rate / (4 * N0))
                        assert math.isclose(m2.rate, 2 * rate / (4 * N0))

        # number of populations doesn't match migration matrix size
        for bad_npop in (1, 20):
            for cmd in (
                f"-I {bad_npop} {' 0'*(bad_npop-1)} 2 -ma 0 0 0 0",
                f"-I {bad_npop} {' 0'*(bad_npop-1)} 2 -ema 0 2 0 0 0 0",
                f"-I {bad_npop} {' 0'*(bad_npop-1)} 2 -ema 10 2 0 0 0 0",
                f"-I 2 2 -ema 0 {bad_npop} {' 0'*bad_npop}",
                f"-I 2 2 -ema 10 {bad_npop} {' 0'*bad_npop}",
            ):
                with pytest.raises(ValueError):
                    demes.from_ms(cmd, N0=1)

        # invalid npop value
        for bad_npop in (-1, 1.1, "x", math.inf):
            for cmd in (
                f"-I 2 2 -ema 0 {bad_npop} 0 0 0 0",
                f"-I 2 2 -ema 10 {bad_npop} 0 0 0 0",
            ):
                with pytest.raises(ValueError):
                    demes.from_ms(cmd, N0=1)

        # invalid population ids
        for bad_popid in (-1, 1.1, "x", math.inf):
            for cmd in (
                f"-I 2 0 2 -m 2 {bad_popid} 0.01",
                f"-I 2 0 2 -m {bad_popid} 2 0.01",
                f"-I 2 0 2 -m {bad_popid} {bad_popid} 0.01",
                f"-I 2 0 2 -em 0 2 {bad_popid} 0.01",
                f"-I 2 0 2 -em 0 {bad_popid} 2 0.01",
                f"-I 2 0 2 -em 0 {bad_popid} {bad_popid} 0.01",
                f"-I 2 0 2 -em 10 2 {bad_popid} 0.01",
                f"-I 2 0 2 -em 10 {bad_popid} 2 0.01",
                f"-I 2 0 2 -em 10 {bad_popid} {bad_popid} 0.01",
                f"-I 10 0 0 0 0 0 0 0 0 2 0 0.4 -m {bad_popid} 7 0.01",
                f"-I 10 0 0 0 0 0 0 0 2 0 0 0.5 -em 0 7 {bad_popid} 0.01",
                f"-I 10 0 0 0 0 0 0 2 0 0 0 0.6 -em 10 7 {bad_popid} 0.01",
            ):
                with pytest.raises(ValueError):
                    demes.from_ms(cmd, N0=1)

        # bad population ids
        for cmd in (
            # ids too high for number of populations
            "-I 2 0 2 -m 2 3 0.01",
            "-I 2 0 2 -m 3 2 0.01",
            "-I 2 0 2 -m 3 4 0.01",
            "-I 2 0 2 -em 0 2 3 0.01",
            "-I 2 0 2 -em 0 3 2 0.01",
            "-I 2 0 2 -em 0 3 4 0.01",
            "-I 2 0 2 -em 10 2 3 0.01",
            "-I 2 0 2 -em 10 3 2 0.01",
            "-I 2 0 2 -em 10 3 4 0.01",
            "-I 10 0 0 0 0 0 0 0 0 2 0 0.4 -m 20 7 0.01",
            "-I 10 0 0 0 0 0 0 0 2 0 0 0.5 -em 0 7 20 0.01",
            "-I 10 0 0 0 0 0 0 2 0 0 0 0.6 -em 10 7 20 0.01",
            # duplicate ids
            "-I 2 0 0 -m 2 2 0.01",
            "-I 2 0 0 -em 0 2 2 0.01",
            "-I 2 0 0 -em 10 2 2 0.01",
        ):
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_time in (-1, math.inf):
            for cmd in (
                f"-I 2 2 -ema {bad_time} 0 0 0 0",
                f"-I 2 2 -ema {bad_time} 0 0.1 0 0",
                f"-I 2 2 -em {bad_time} 1 2 0",
                f"-I 2 2 -em {bad_time} 1 2 0.1",
            ):
                with pytest.raises(ValueError):
                    demes.from_ms(cmd, N0=1)

        for bad_rate in (-1, math.inf):
            for cmd in (
                f"-I 2 2 -m 1 2 {bad_rate}",
                f"-I 2 2 -em 0 1 2 {bad_rate}",
                f"-I 2 2 -em 10 1 2 {bad_rate}",
                f"-I 2 0 2 -ma 0 {bad_rate} 0 0",
                f"-I 2 0 2 -ema 0 2 0 {bad_rate} 0 0",
                f"-I 2 0 2 -ema 10 2 0 {bad_rate} 0 0",
            ):
                with pytest.raises(ValueError):
                    demes.from_ms(cmd, N0=1)

    def test_asymmetric_migration(self):
        # In ms, M[i][j] is the (forwards time) fraction of deme i which is
        # made up of migrants from deme j each generation.
        # See ./ms_asymmetric_migration.sh for a test that confirms this
        # using the ms commands below.

        def check(cmd):
            graph = demes.from_ms(cmd, N0=1000)
            assert len(graph.demes) == 3
            assert len(graph.migrations) == 1
            assert graph.migrations[0].source == "deme3"
            assert graph.migrations[0].dest == "deme2"

        # Set M[3][2] = 1.0 using the -em option.
        cmd1 = (
            "ms 2 1000 "
            "-t 1.0 "
            "-I 3 1 0 1 "
            "-em 0.0 3 2 1.0 "
            "-ej 100.0 2 1 "
            "-eM 100.0 0.0"
        )
        check(cmd1)

        # Set M[3][2] = 1.0 using the -ema option
        cmd2 = (
            "ms 2 1000 "
            "-t 1.0 "
            "-I 3 1 0 1 "
            "-ema 0.0 3 x 0 0 0 x 0 0 1.0 x "
            "-ej 100.0 2 1 "
            "-eM 100.0 0.0"
        )
        check(cmd2)

        # Set M[3][2] = 1.0 using the -ma option
        cmd3 = (
            "ms 2 1000 "
            "-t 1.0 "
            "-I 3 1 0 1 "
            "-ma x 0 0 0 x 0 0 1.0 x "
            "-ej 100.0 2 1 "
            "-eM 100.0 0.0"
        )
        check(cmd3)

        # Set M[3][2] = 1.0 using the -m option.
        cmd4 = (
            "ms 2 1000 "
            "-t 1.0 "
            "-I 3 1 0 1 "
            "-m 3 2 1.0 "
            "-ej 100.0 2 1 "
            "-eM 100.0 0.0"
        )
        check(cmd4)

    def test_admixture(self):
        # -es t i p
        # from_ms() creates a new deme (say, j) with end_time=t,
        # and a pulse of migration from deme j into deme i at time t.
        N0 = 100
        for t in (0.01, 1, 10):
            for p in (0.01, 0.5, 0.9):
                for num_demes, dest_id in (
                    (1, 1),
                    (2, 2),
                    (10, 3),
                ):
                    cmd = (
                        f"-I {num_demes} {' 0'*(num_demes - 1)} 2 "
                        "-eN 0 12 "
                        f"-es {t} {dest_id} {p}"
                    )
                    graph = demes.from_ms(cmd, N0=N0)
                    assert len(graph.demes) == num_demes + 1
                    assert len(graph.migrations) == 0
                    assert len(graph.pulses) == 1
                    dest = f"deme{dest_id}"
                    source = f"deme{num_demes + 1}"
                    # pulse properties
                    pulse = graph.pulses[0]
                    assert pulse.source == source
                    assert pulse.dest == dest
                    assert math.isclose(pulse.proportion, 1 - p)
                    assert math.isclose(pulse.time, t * 4 * N0)
                    # source deme properties
                    assert math.isinf(graph[source].start_time)
                    assert math.isclose(graph[source].end_time, t * 4 * N0)
                    assert len(graph[source].epochs) == 1
                    assert math.isclose(graph[source].epochs[0].start_size, N0)
                    assert math.isclose(graph[source].epochs[0].end_size, N0)
                    # other demes
                    for deme in graph.demes:
                        if deme.name == source:
                            continue
                        assert math.isinf(deme.start_time)
                        assert deme.end_time == 0
                        assert len(deme.epochs) == 1
                        assert math.isclose(deme.epochs[0].start_size, 12 * N0)
                        assert math.isclose(deme.epochs[0].end_size, 12 * N0)

        for bad_t in (-1, 0, math.inf):
            cmd = f"-es {bad_t} 1 0.1"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_popid in (-1, 200, math.inf):
            cmd = f"-es 1.0 {bad_popid} 0.1"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_p in (-1, 2, math.inf):
            cmd = f"-es 1.0 1 {bad_p}"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

    def test_split(self):
        # -ej t i j
        # from_ms() turns this into a "branch" event where
        # deme i branches from deme j at time t.
        N0 = 100
        for t in (0.01, 1, 10):
            for num_demes, i, j in (
                (2, 1, 2),
                (2, 2, 1),
                (10, 3, 8),
            ):
                cmd = (
                    f"-I {num_demes} {' 0'*(num_demes - 1)} 2 1.0 "
                    "-eN 0 12 "
                    f"-ej {t} {i} {j}"
                )
                graph = demes.from_ms(cmd, N0=N0)
                assert len(graph.demes) == num_demes
                assert len(graph.migrations) == num_demes * (num_demes - 1)
                assert len(graph.pulses) == 0
                child = f"deme{i}"
                assert math.isclose(graph[child].start_time, t * 4 * N0)
                assert graph[child].end_time == 0
                assert len(graph[child].epochs) == 1
                assert math.isclose(graph[child].epochs[0].start_size, 12 * N0)
                assert math.isclose(graph[child].epochs[0].end_size, 12 * N0)
                for deme in graph.demes:
                    if deme.name == child:
                        continue
                    assert math.isinf(deme.start_time)
                    assert deme.end_time == 0
                    assert len(deme.epochs) == 1
                    assert math.isclose(deme.epochs[0].start_size, 12 * N0)
                    assert math.isclose(deme.epochs[0].end_size, 12 * N0)
                # check migrations
                rate = 1 / (4 * N0)
                for migration in graph.migrations:
                    assert math.isclose(migration.rate, rate / (num_demes - 1))
                    # migrations should match the start/end time of the participants
                    start_time = min(
                        graph[migration.source].start_time,
                        graph[migration.dest].start_time,
                    )
                    end_time = max(
                        graph[migration.source].end_time, graph[migration.dest].end_time
                    )
                    assert start_time == migration.start_time
                    assert end_time == migration.end_time

        # check a model with successive branching
        T1, T2, T3, T4 = 1.0, 2.0, 3.0, 4.0
        cmd = (
            "-I 5 1 1 1 1 1 "
            f"-ej {T1} 1 2 "
            f"-ej {T2} 2 3 "
            f"-ej {T3} 3 4 "
            f"-ej {T4} 4 5 "
        )
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 5
        assert len(graph.pulses) == 0
        assert len(graph.migrations) == 0
        assert len([deme for deme in graph.demes if math.isinf(deme.start_time)])
        assert [t * 4 * N0 for t in [T1, T2, T3, T4, math.inf]] == sorted(
            [deme.start_time for deme in graph.demes]
        )
        assert [0, 0, 0, 0, 0] == [deme.end_time for deme in graph.demes]
        assert [1, 1, 1, 1, 1] == [len(deme.epochs) for deme in graph.demes]
        assert graph["deme1"].ancestors == ["deme2"]
        assert graph["deme2"].ancestors == ["deme3"]
        assert graph["deme3"].ancestors == ["deme4"]
        assert graph["deme4"].ancestors == ["deme5"]
        assert graph["deme5"].ancestors == []

        for bad_t in (-1, 0, math.inf):
            cmd = f"-I 2 1 1 -ej {bad_t} 1 2"
            with pytest.raises(ValueError):
                demes.from_ms(cmd, N0=1)

        for bad_cmd in (
            # deme doesn't exist
            "-ej 1.0 1 2",
            "-ej 1.0 2 1",
            "-ej 1.0 2 3",
            "-I 3 0 0 2 -ej 1.0 1 4",
            "-I 3 0 0 2 -ej 1.0 4 1",
            "-I 3 0 0 2 -ej 1.0 4 10",
            # can't split a deme from itself
            "-ej 1.0 1 1",
            "-I 3 0 0 2 -ej 1.0 3 3",
            # can't split a deme after it's already split
            "-I 3 0 0 2 -ej 1.0 1 2 -ej 2.0 1 2",
            "-I 3 0 0 2 -ej 1.0 1 2 -ej 2.0 1 3",
        ):
            with pytest.raises(ValueError):
                demes.from_ms(bad_cmd, N0=1)

    def test_admixture_and_split(self):
        # Test that -ej and -es work together.
        N0 = 100
        T1, T2 = 1.0, 2.0
        p = 0.1
        cmd = f"-es {T1} 1 {p} -ej {T2} 2 1"
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 2
        assert len(graph.pulses) == 1
        assert len(graph.migrations) == 0
        # check deme1
        assert graph["deme1"].ancestors == []
        assert math.isinf(graph["deme1"].start_time)
        assert graph["deme1"].end_time == 0
        assert len(graph["deme1"].epochs) == 1
        # check deme2
        assert graph["deme2"].ancestors == ["deme1"]
        assert math.isclose(graph["deme2"].start_time, T2 * 4 * N0)
        assert math.isclose(graph["deme2"].end_time, T1 * 4 * N0)
        assert len(graph["deme2"].epochs) == 1
        # check pulse
        pulse = graph.pulses[0]
        assert pulse.source == "deme2"
        assert pulse.dest == "deme1"
        assert pulse.proportion == 1 - p
        assert math.isclose(pulse.time, T1 * 4 * N0)

    def test_args_from_file(self):
        # -f filename

        def writeparams(filename, params):
            with open(filename, "w") as f:
                print(params, file=f)

        def check_im(graph, *, num_demes, M, N0):
            assert len(graph.demes) == num_demes
            assert len(graph.migrations) == num_demes * (num_demes - 1)
            rate = M / (4 * N0)
            for migration in graph.migrations:
                assert math.isclose(migration.rate, rate / (num_demes - 1))

        def check_epochs(graph, *, alpha, T, N0):
            for deme in graph.demes:
                assert len(deme.epochs) == 2
                e1, e2 = deme.epochs
                size = N0 * math.exp(-alpha * T)
                assert math.isinf(e1.start_time)
                assert math.isclose(e1.end_time, T * 4 * N0)
                assert math.isclose(e2.start_time, T * 4 * N0)
                assert math.isclose(e2.end_time, 0)
                assert math.isclose(e1.start_size, size)
                assert math.isclose(e1.end_size, size)
                assert math.isclose(e2.start_size, size)
                assert math.isclose(e2.end_size, N0)

        N0 = 100
        T = 1.0
        M = 0.1
        alpha = 0.01
        with tempfile.TemporaryDirectory() as tmpdir:
            paramfile = str(pathlib.Path(tmpdir) / "params.txt")
            writeparams(paramfile, f"-I 2 1 1 {M}")
            # param file on its own
            cmd = f"-f {paramfile}"
            graph = demes.from_ms(cmd, N0=N0)
            check_im(graph, num_demes=2, M=M, N0=N0)
            # other params before -f
            cmd = f"-eG 0 {alpha} -eG {T} 0 -f {paramfile}"
            graph = demes.from_ms(cmd, N0=N0)
            check_im(graph, num_demes=2, M=M, N0=N0)
            check_epochs(graph, alpha=alpha, T=T, N0=N0)
            # params after -f
            cmd = f"-f {paramfile} -eG 0 {alpha} -eG {T} 0 "
            graph = demes.from_ms(cmd, N0=N0)
            check_im(graph, num_demes=2, M=M, N0=N0)
            check_epochs(graph, alpha=alpha, T=T, N0=N0)
            # params before and after -f
            cmd = f"-eG 0 {alpha} -f {paramfile} -eG {T} 0 "
            graph = demes.from_ms(cmd, N0=N0)
            check_im(graph, num_demes=2, M=M, N0=N0)
            check_epochs(graph, alpha=alpha, T=T, N0=N0)
            # -f inside a file specified with -f
            paramfile2 = str(pathlib.Path(tmpdir) / "params2.txt")
            writeparams(paramfile2, f"-eG 0 {alpha} -eG {T} 0 -f {paramfile}")
            cmd = f"-f {paramfile2}"
            graph = demes.from_ms(cmd, N0=N0)
            check_im(graph, num_demes=2, M=M, N0=N0)
            check_epochs(graph, alpha=alpha, T=T, N0=N0)

        # nonexistent file
        cmd = "-f nonexistent"
        with pytest.raises(OSError):
            demes.from_ms(cmd, N0=1)

    def test_bad_arguments(self):
        # Check that giving the wrong number of options is an error.
        for bad_cmd in (
            "-f",
            "-G",
            "-I",
            "-I 2 1 1 -g",
            "-I 2 1 1 -m",
            "-I 2 1 1 -ma",
            "-I 2 1 1 -eG",
            "-I 2 1 1 -eg",
            "-I 2 1 1 -eN",
            "-I 2 1 1 -en",
            "-I 2 1 1 -eM",
            "-I 2 1 1 -em",
            "-I 2 1 1 -ema",
            "-I 2 1 1 -es",
            "-I 2 1 1 -ej",
        ):
            with pytest.raises(ValueError):
                demes.from_ms(bad_cmd, N0=1)

    ##
    # Examples from the ms manual (November 10 2018).

    def test_msdoc_example1(self):
        # Instantaneous population size changes.
        cmd = "ms 15 1000 -t 2.0 -eN 1.0 .1 -eN 2.0 4.0"
        graph = demes.from_ms(cmd, N0=1)
        assert len(graph.migrations) == 0
        assert len(graph.pulses) == 0
        assert len(graph.demes) == 1
        assert len(graph.demes[0].epochs) == 3
        e1, e2, e3 = graph.demes[0].epochs
        assert e1.start_size == e1.end_size
        assert e2.start_size == e2.end_size
        assert e3.start_size == e3.end_size
        assert math.isclose(e2.end_size, 0.1 * e3.end_size)
        assert math.isclose(e1.end_size, 4 * e3.end_size)

    def test_msdoc_example2(self):
        # Generating an outgroup sequence.
        cmd = "ms 11 3 -t 2.0 -I 2 1 10 -ej 6.0 1 2"
        N0 = self.N0(theta=2, mu=1, length=1)
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.migrations) == 0
        assert len(graph.pulses) == 0
        assert len(graph.demes) == 2
        d1, d2 = graph["deme1"], graph["deme2"]
        assert len(d1.epochs) == 1
        assert len(d2.epochs) == 1
        assert len(d1.ancestors) == 1
        assert len(d2.ancestors) == 0
        assert d2.name in d1.ancestors
        assert math.isclose(24 * N0, d1.start_time)

    def test_msdoc_example3(self):
        # Instantaneous size change followed by exponential growth.
        # Figure 1.
        cmd = "ms 15 1000 -t 6.4 -G 6.93 -eG 0.2 0.0 -eN 0.3 0.5"
        graph = demes.from_ms(cmd, N0=self.N0(theta=6.4, mu=1e-8, length=8000))
        assert len(graph.migrations) == 0
        assert len(graph.pulses) == 0
        assert len(graph.demes) == 1
        assert len(graph.demes[0].epochs) == 3

        N1, N2, N3 = 10000, 5000, 20000
        T1, T2 = 16000, 24000
        e1, e2, e3 = graph.demes[0].epochs
        assert math.isclose(e1.start_size, N1)
        assert math.isclose(e1.end_size, N1)
        # Note: coarse rounding used for the growth rate in the manual.
        assert math.isclose(e2.start_size, N2, rel_tol=1e-3)
        assert math.isclose(e2.end_size, N2, rel_tol=1e-2)
        assert math.isclose(e3.start_size, N2, rel_tol=1e-3)
        assert math.isclose(e3.end_size, N3)
        assert math.isclose(e1.end_time, T2)
        assert math.isclose(e2.end_time, T1)
        assert math.isclose(e3.end_time, 0)

    def test_msdoc_example4(self):
        # Two species with population size differences.
        # Figure 2.

        def verify(graph):
            N1, N2, N3, N4 = 10000, 2000, 5000, 40000
            T1, T2, T3 = 5000, 10000, 15000
            rel_tol = 1e-4

            assert len(graph.migrations) == 0
            assert len(graph.pulses) == 0
            assert len(graph.demes) == 2
            d1, d2 = graph["deme1"], graph["deme2"]
            assert math.isinf(d1.start_time)
            assert d1.end_time == 0
            assert math.isclose(d2.start_time, T3)
            assert d2.end_time == 0

            assert len(d1.epochs) == 2
            d1e1, d1e2 = d1.epochs
            assert math.isclose(d1e1.start_size, N1, rel_tol=rel_tol)
            assert math.isclose(d1e1.end_size, N1, rel_tol=rel_tol)
            assert math.isclose(d1e2.start_size, N1, rel_tol=rel_tol)
            assert math.isclose(d1e2.end_size, N4, rel_tol=rel_tol)
            assert math.isinf(d1e1.start_time)
            assert math.isclose(d1e1.end_time, T1)
            assert math.isclose(d1e2.start_time, T1)
            assert math.isclose(d1e2.end_time, 0)

            assert len(d2.epochs) == 2
            d2e1, d2e2 = d2.epochs
            assert math.isclose(d2e1.start_size, N2, rel_tol=rel_tol)
            assert math.isclose(d2e1.end_size, N2, rel_tol=rel_tol)
            assert math.isclose(d2e2.start_size, N3, rel_tol=rel_tol)
            assert math.isclose(d2e2.end_size, N3, rel_tol=rel_tol)
            assert math.isclose(d2e1.start_time, T3)
            assert math.isclose(d2e1.end_time, T2)
            assert math.isclose(d2e2.start_time, T2)
            assert math.isclose(d2e2.end_time, 0)

        mu = 1e-8
        sequence_length = 7000
        cmd1 = (
            "ms 15 100 "
            "-t 11.2 "
            "-I 2 3 12 "
            "-g 1 44.36 "
            "-n 2 0.125 "
            "-eg 0.03125 1 0.0 "
            "-en 0.0625 2 0.05 "
            "-ej 0.09375 2 1"
        )
        graph1 = demes.from_ms(
            cmd1, N0=self.N0(theta=11.2, mu=mu, length=sequence_length)
        )
        verify(graph1)

        # A second command is provided in the manual, using a different N0,
        # which should result in an identical model.
        cmd2 = (
            "ms 15 100 "
            "-t 2.8 "
            "-I 2 3 12 "
            "-g 1 11.09 "
            "-n 1 4.0 "
            "-n 2 0.5 "
            "-eg 0.125 1 0.0 "
            "-en 0.25 2 .2 "
            "-ej 0.375 2 1"
        )
        graph2 = demes.from_ms(
            cmd2, N0=self.N0(theta=2.8, mu=mu, length=sequence_length)
        )
        verify(graph2)

        graph1.assert_close(graph2)

    def test_msdoc_example5(self):
        # Stepping stone model with recent barrier.
        # Figure 3.
        cmd = (
            "ms 15 100 "
            "-t 3.0 "
            "-I 6 0 7 0 0 8 0 "
            "-m 1 2 2.5 "
            "-m 2 1 2.5 "
            "-m 2 3 2.5 "
            "-m 3 2 2.5 "
            "-m 4 5 2.5 "
            "-m 5 4 2.5 "
            "-m 5 6 2.5 "
            "-m 6 5 2.5 "
            "-em 2.0 3 4 2.5 "
            "-em 2.0 4 3 2.5"
        )
        N0 = self.N0(theta=3, mu=1e-8, length=1)
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.pulses) == 0
        assert len(graph.migrations) == 10
        assert len(graph.demes) == 6

        def get_migrations(graph, source, dest):
            """Return a list of the migrations from source to dest."""
            migrations = []
            for migration in graph.migrations:
                if migration.source == source and migration.dest == dest:
                    migrations.append(migration)
            return migrations

        migration_rate = 2.5 / (4 * N0)
        # time of migration barrier
        T1 = 2.0 * 4 * N0

        for d1, d2 in [
            ("deme1", "deme2"),
            ("deme2", "deme3"),
            ("deme4", "deme5"),
            ("deme5", "deme6"),
        ]:
            m12 = get_migrations(graph, d1, d2)
            m21 = get_migrations(graph, d2, d1)
            assert len(m12) == len(m21) == 1
            m12 = m12[0]
            m21 = m21[0]
            assert math.isclose(m12.rate, migration_rate)
            assert math.isclose(m21.rate, migration_rate)
            assert math.isinf(m12.start_time)
            assert math.isinf(m21.start_time)
            assert math.isclose(m12.end_time, 0)
            assert math.isclose(m21.end_time, 0)

        m34 = get_migrations(graph, "deme3", "deme4")
        m43 = get_migrations(graph, "deme4", "deme3")
        assert len(m34) == len(m43) == 1
        m34 = m34[0]
        m43 = m43[0]
        assert math.isclose(m34.rate, migration_rate)
        assert math.isclose(m43.rate, migration_rate)
        assert math.isinf(m34.start_time)
        assert math.isinf(m43.start_time)
        assert math.isclose(m34.end_time, T1)
        assert math.isclose(m43.end_time, T1)

    ##
    # Ms commands from publications.

    def test_zigzag(self):
        # Schiffels & Durbin (2014), https://doi.org/10.1038/ng.3015
        # Ms command from section 7 of the supplement (pg. 25).
        cmd = (
            "ms 4 1 "
            "-t 7156.0000000 "
            "-r 2000.0000 10000000 "
            "-eN 0 5 "
            "-eG 0.000582262 1318.18 "
            "-eG 0.00232905 -329.546 "
            "-eG 0.00931619 82.3865 "
            "-eG 0.0372648 -20.5966 "
            "-eG 0.149059 5.14916 "
            "-eN 0.596236 0.5 "
            "-T"
        )
        mu = 1.25e-8
        theta = 7156
        sequence_length = 10000000
        graph = demes.from_ms(
            cmd, N0=self.N0(theta=theta, mu=mu, length=sequence_length)
        )
        assert len(graph.migrations) == 0
        assert len(graph.pulses) == 0
        assert len(graph.demes) == 1
        assert len(graph.demes[0].epochs) == 7

        # Compare with zigzag in examples folder.
        zz = demes.load(example_dir / "zigzag.yml")
        zz.generation_time = None
        zz.demes[0].name = "deme1"
        # We use a lower tolerance than the default because the example model
        # uses somewhat arbitrary rounding.
        graph.assert_close(zz, rel_tol=1e-4)

    def test_gutenkunst_ooa(self):
        # Gutenkunst et al. (2009), https://doi.org/10.1371/journal.pgen.1000695
        # Ms command from section 5.2 of the supplement (pg. 10--12).
        cmd = (
            "-I 3 1 1 1 "
            "-n 1 1.682020 "
            "-n 2 3.736830 "
            "-n 3 7.292050 "
            "-eg 0 2 116.010723 "
            "-eg 0 3 160.246047 "
            "-ma x 0.881098 0.561966 0.881098 x 2.797460 0.561966 2.797460 x "
            "-ej 0.028985 3 2 "
            "-en 0.028985 2 0.287184 "
            "-ema 0.028985 3 x 7.293140 x 7.293140 x x x x x "
            "-ej 0.197963 2 1 "
            "-en 0.303501 1 1"
        )
        graph = demes.from_ms(
            cmd,
            N0=7310,
            deme_names=["YRI", "CEU", "CHB"],
        )
        assert len(graph.demes) == 3
        assert len(graph.migrations) == 8
        # TODO: more checks.
        # We don't compare to the gutenkunst_ooa.yml file in the examples
        # folder because that graph introduces additional demes for the
        # ancestral and OOA populations which are not present in the original
        # ms-based demographic model.

    ##
    # api stuff

    def test_deme_names(self):
        graph = demes.from_ms("-I 2 1 1 1.0", N0=1, deme_names=["A", "B"])
        assert len(graph.demes) == 2
        assert "A" in graph
        assert "B" in graph
        for migration in graph.migrations:
            assert migration.source in ["A", "B"]
            assert migration.dest in ["A", "B"]

        graph = demes.from_ms("-es 1.0 1 0.1", N0=1, deme_names=["A", "B"])
        assert len(graph.demes) == 2
        assert "A" in graph
        assert "B" in graph
        assert len(graph.pulses) == 1
        assert graph.pulses[0].source == "B"
        assert graph.pulses[0].dest == "A"

        # bad deme names
        for bad_deme_names in (
            ["A"],
            ["A", "A"],
            ["A", "B", "C"],
        ):
            with pytest.raises(ValueError):
                demes.from_ms("-I 2 1 1", N0=1, deme_names=bad_deme_names)

    def test_unhandled_event(self):
        from demes.ms import parse_ms_args, build_graph

        class Foo:
            t: float

        foo = Foo()
        foo.t = 0
        args, _ = parse_ms_args("-I 2 1 1")
        args.initial_state.append(foo)
        with pytest.raises(AssertionError):
            build_graph(args, 1)

        foo.t = 1.0
        args, _ = parse_ms_args("-I 2 1 1")
        args.initial_state.append(foo)
