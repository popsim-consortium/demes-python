import collections
import math
import pathlib
import pytest
import tempfile

import demes
from demes import ms
import tests


def N_ref(*, theta, mu, length):
    """
    Calculate N0 reference size for an ms command.
    """
    return theta / (4 * mu * length)


def N_ref_macs(*, theta, mu):
    """
    Calculate N0 reference size for a macs command.
    Macs has a different meaning for the '-t theta' argument.
    Theta is the 'mutation rate per site per 4N generations'.
    theta = mu / (4 N)
    """
    return theta / (4 * mu)


class TestFromMs:
    def test_ignored_options_have_no_effect(self):
        def check(command, N0=1):
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
            cmd = f"ms 2 1 -t 1.0 -I {bad_sample_configuration}"
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
            assert graph.migrations[0].source == "deme2"
            assert graph.migrations[0].dest == "deme3"

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

    def test_split(self):
        # -es t i p
        # This is an admixture event forwards in time.
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
                    assert pulse.sources[0] == source
                    assert pulse.dest == dest
                    assert math.isclose(pulse.proportions[0], 1 - p)
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

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_split_twice_immediately(self):
        # This represents a deme with multiple ancestors.
        N0 = 1
        T = 1.0
        graph = demes.from_ms(f"-es {T} 1 0.7 -es {T} 1 0.8", N0=N0)
        assert len(graph.demes) == 3
        assert math.isinf(graph["deme1"].start_time)
        assert graph["deme1"].end_time == 0
        assert math.isclose(graph["deme2"].end_time, T * 4 * N0)
        assert math.isclose(graph["deme3"].end_time, T * 4 * N0)
        assert len(graph.pulses) == 2
        # The order of pulses matters here.
        assert graph.pulses[0].sources[0] == "deme3"
        assert graph.pulses[0].dest == "deme1"
        assert math.isclose(graph.pulses[0].proportions[0], 1 - 0.8)
        assert math.isclose(graph.pulses[0].time, T * 4 * N0)
        assert graph.pulses[1].sources[0] == "deme2"
        assert graph.pulses[1].dest == "deme1"
        assert math.isclose(graph.pulses[1].proportions[0], 1 - 0.7)
        assert math.isclose(graph.pulses[1].time, T * 4 * N0)

    def test_join(self):
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
            # can't join a deme with itself
            "-ej 1.0 1 1",
            "-I 3 0 0 2 -ej 1.0 3 3",
            # can't join a deme after it's already joined
            "-I 3 0 0 2 -ej 1.0 1 2 -ej 2.0 1 2",
            "-I 3 0 0 2 -ej 1.0 1 2 -ej 2.0 1 3",
        ):
            with pytest.raises(ValueError):
                demes.from_ms(bad_cmd, N0=1)

    def test_join_then_join_again_immediately(self):
        N0 = 100
        T1 = 1.0
        cmd = f"-I 3 2 2 2 -ej {T1} 3 2 -ej {T1} 2 1"
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 3
        assert graph["deme1"].ancestors == []
        assert graph["deme2"].ancestors == ["deme1"]
        assert graph["deme3"].ancestors == ["deme1"]

    def test_set_size_after_join(self):
        # Set size of all demes after one deme has been joined.
        N0 = 100
        T1, T2 = 1.0, 2.0
        cmd = f"-I 2 1 1 -ej {T1} 2 1 -eN {T2} 2"
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 2
        assert len(graph.pulses) == 0
        assert len(graph.migrations) == 0
        # check deme1
        assert math.isinf(graph["deme1"].start_time)
        assert graph["deme1"].end_time == 0
        assert len(graph["deme1"].epochs) == 2
        assert math.isclose(graph["deme1"].epochs[0].start_size, N0 * 2)
        assert math.isclose(graph["deme1"].epochs[1].start_size, N0 * 1)
        # check deme2
        assert math.isclose(graph["deme2"].start_time, 4 * N0 * T1)
        assert graph["deme2"].end_time == 0
        assert len(graph["deme2"].epochs) == 1
        assert math.isclose(graph["deme2"].epochs[0].start_size, N0 * 1)

    def test_set_growth_rate_after_join(self):
        # Set growth rate of all demes after one deme has been joined.
        N0 = 100
        T1, T2 = 1.0, 2.0
        alpha = -0.01
        cmd = f"-I 2 1 1 -G {alpha} -ej {T1} 2 1 -eG {T2} 0"
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 2
        assert len(graph.pulses) == 0
        assert len(graph.migrations) == 0
        # check deme1
        assert math.isinf(graph["deme1"].start_time)
        assert graph["deme1"].end_time == 0
        assert len(graph["deme1"].epochs) == 2
        size_at_T2 = N0 * math.exp(-alpha * T2)
        assert math.isclose(graph["deme1"].epochs[0].start_size, size_at_T2)
        assert math.isclose(graph["deme1"].epochs[0].end_size, size_at_T2)
        assert math.isclose(graph["deme1"].epochs[1].start_size, size_at_T2)
        assert math.isclose(graph["deme1"].epochs[1].end_size, N0 * 1)
        # check deme2
        assert math.isclose(graph["deme2"].start_time, 4 * N0 * T1)
        assert graph["deme2"].end_time == 0
        assert len(graph["deme2"].epochs) == 1
        size_at_T1 = N0 * math.exp(-alpha * T1)
        assert math.isclose(graph["deme2"].epochs[0].start_size, size_at_T1)
        assert math.isclose(graph["deme2"].epochs[0].end_size, N0 * 1)

    def test_set_migration_rate_after_join(self):
        # Set migration rate between all demes after one deme has been joined.
        N0 = 100
        T1, T2 = 1.0, 2.0
        cmd = f"-I 3 2 2 2 1.0 -ej {T1} 3 2 -eM {T2} 0"
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 3
        assert len(graph.pulses) == 0
        assert len(graph.migrations) == 6
        m = 1.0 / (4 * N0 * (len(graph.demes) - 1))  # rate for island model
        start_times = {}
        for migration in graph.migrations:
            assert math.isclose(migration.rate, m)
            assert migration.end_time == 0
            start_times[(migration.source, migration.dest)] = migration.start_time
        assert len(start_times) == len(graph.migrations)
        assert math.isclose(start_times[("deme1", "deme2")], T2 * 4 * N0)
        assert math.isclose(start_times[("deme2", "deme1")], T2 * 4 * N0)
        assert math.isclose(start_times[("deme1", "deme3")], T1 * 4 * N0)
        assert math.isclose(start_times[("deme3", "deme1")], T1 * 4 * N0)
        assert math.isclose(start_times[("deme2", "deme3")], T1 * 4 * N0)
        assert math.isclose(start_times[("deme3", "deme2")], T1 * 4 * N0)

    def test_split_then_join(self):
        # Test that -es and -ej work together.
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
        assert pulse.sources[0] == "deme2"
        assert pulse.dest == "deme1"
        assert pulse.proportions[0] == 1 - p
        assert math.isclose(pulse.time, T1 * 4 * N0)

    def test_split_then_join_immediately(self):
        # Test that -es followed immediately by -ej works.
        N0 = 100
        T1 = 1.0
        p = 0.1
        cmd = f"-I 2 1 1 0.5 -es {T1} 1 {p} -ej {T1} 3 2"
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 2
        assert len(graph.pulses) == 1
        # check deme1
        assert graph["deme1"].ancestors == []
        assert math.isinf(graph["deme1"].start_time)
        assert graph["deme1"].end_time == 0
        assert len(graph["deme1"].epochs) == 1
        # check deme2
        assert graph["deme2"].ancestors == []
        assert math.isinf(graph["deme2"].start_time)
        assert graph["deme2"].end_time == 0
        assert len(graph["deme2"].epochs) == 1
        # check pulse
        pulse = graph.pulses[0]
        assert pulse.sources[0] == "deme2"
        assert pulse.dest == "deme1"
        assert pulse.proportions[0] == 1 - p
        assert math.isclose(pulse.time, T1 * 4 * N0)

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_split_then_join_sequence_1(self):
        # backwards in time we have: deme3 -> deme2; deme2 -> deme1
        N0 = 100
        T1 = 1.0
        cmd = (
            "-I 3 1 1 1 0.5 "
            f"-es {T1} 3 0.6 -ej {T1} 4 2 "
            f"-es {T1} 2 0.6 -ej {T1} 5 1 "
        )
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 3
        assert len(graph.pulses) == 2
        assert graph.pulses[0].sources[0] == "deme1"
        assert graph.pulses[0].dest == "deme2"
        assert math.isclose(graph.pulses[0].time, T1 * 4 * N0)
        assert math.isclose(graph.pulses[0].proportions[0], 0.4)
        assert graph.pulses[1].sources[0] == "deme2"
        assert graph.pulses[1].dest == "deme3"
        assert math.isclose(graph.pulses[1].time, T1 * 4 * N0)
        assert math.isclose(graph.pulses[1].proportions[0], 0.4)

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_split_then_join_sequence_2(self):
        # backwards in time we have: deme3 -> deme2; deme3 -> deme1
        N0 = 100
        T1 = 1.0
        cmd = (
            "-I 3 1 1 1 0.5 "
            f"-es {T1} 3 0.6 -ej {T1} 4 2 "
            f"-es {T1} 3 0.6 -ej {T1} 5 1 "
        )
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 3
        assert len(graph.pulses) == 2
        assert graph.pulses[0].sources[0] == "deme1"
        assert graph.pulses[0].dest == "deme3"
        assert math.isclose(graph.pulses[0].time, T1 * 4 * N0)
        assert math.isclose(graph.pulses[0].proportions[0], 0.4)
        assert graph.pulses[1].sources[0] == "deme2"
        assert graph.pulses[1].dest == "deme3"
        assert math.isclose(graph.pulses[1].time, T1 * 4 * N0)
        assert math.isclose(graph.pulses[1].proportions[0], 0.4)

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_split_then_join_sequence_3(self):
        # backwards in time we have: deme3 -> deme2; deme1 -> deme2
        N0 = 100
        T1 = 1.0
        cmd = (
            "-I 3 1 1 1 0.5 "
            f"-es {T1} 3 0.6 -ej {T1} 4 2 "
            f"-es {T1} 1 0.6 -ej {T1} 5 2 "
        )
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 3
        assert len(graph.pulses) == 2
        assert graph.pulses[0].sources[0] == "deme2"
        assert graph.pulses[0].dest == "deme1"
        assert math.isclose(graph.pulses[0].time, T1 * 4 * N0)
        assert math.isclose(graph.pulses[0].proportions[0], 0.4)
        assert graph.pulses[1].sources[0] == "deme2"
        assert graph.pulses[1].dest == "deme3"
        assert math.isclose(graph.pulses[1].time, T1 * 4 * N0)
        assert math.isclose(graph.pulses[1].proportions[0], 0.4)

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_split_then_join_sequence_then_join(self):
        # This is how one might specify multiple ancestors.
        # deme4 has ancestry from each of deme1, deme2, deme3,
        # and none of the ancestors go extinct when deme4 is created.
        N0 = 100
        T1 = 1.0
        cmd = (
            "-I 4 1 1 1 1 0.5 "
            f"-es {T1} 4 0.6 -ej {T1} 5 1 "
            f"-es {T1} 4 0.6 -ej {T1} 6 2 "
            f"-ej {T1} 4 3"
        )
        graph = demes.from_ms(cmd, N0=N0)
        assert len(graph.demes) == 4
        assert math.isclose(graph["deme4"].start_time, T1 * 4 * N0)
        assert graph["deme4"].end_time == 0
        for deme_name in ("deme1", "deme2", "deme3"):
            deme = graph[deme_name]
            assert math.isinf(deme.start_time)
            assert deme.end_time == 0

        # Two possible resolutions make sense here:
        # 1) deme4 has one ancestor, and there are two additional pulses of
        #    ancestry into deme4,
        # assert len(graph["deme4"].ancestors) == 1 and len(graph.pulses) == 2
        # 2) alternately, deme4 has three ancestors, and there are no pulses.
        #    This is nicer, but more difficult to implement.
        ancestors, proportions = zip(
            *sorted(zip(graph["deme4"].ancestors, graph["deme4"].proportions))
        )
        assert ancestors == ("deme1", "deme2", "deme3")
        assert math.isclose(proportions[0], 0.4)  # deme1
        assert math.isclose(proportions[1], 0.6 * 0.4)  # deme2
        assert math.isclose(proportions[2], 1 - 0.4 - 0.6 * 0.4)  # deme3
        assert len(graph.pulses) == 0

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
        N0 = N_ref(theta=2, mu=1, length=1)
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
        graph = demes.from_ms(cmd, N0=N_ref(theta=6.4, mu=1e-8, length=8000))
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
            cmd1, N0=N_ref(theta=11.2, mu=mu, length=sequence_length)
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
        graph2 = demes.from_ms(cmd2, N0=N_ref(theta=2.8, mu=mu, length=sequence_length))
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
        N0 = N_ref(theta=3, mu=1e-8, length=1)
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
        graph = demes.from_ms(cmd, N0=N_ref(theta=theta, mu=mu, length=sequence_length))
        assert len(graph.migrations) == 0
        assert len(graph.pulses) == 0
        assert len(graph.demes) == 1
        assert len(graph.demes[0].epochs) == 7

        # Compare with zigzag in examples folder.
        zz = demes.load(tests.example_dir / "zigzag.yaml")
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
        # We don't compare to the gutenkunst_ooa.yaml file in the examples
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
        assert graph.pulses[0].sources[0] == "B"
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
        from demes.ms import build_parser, build_graph

        parser = build_parser()

        class Foo:
            t: float

        foo = Foo()
        foo.t = 0
        args = parser.parse_args("-I 2 1 1".split())
        args.initial_state.append(foo)
        with pytest.raises(AssertionError):
            build_graph(args, 1)

        foo.t = 1.0
        args = parser.parse_args("-I 2 1 1".split())
        args.demographic_events.append(foo)
        with pytest.raises(AssertionError):
            build_graph(args, 1)


# Less stringent tests for a broader range of models seen in the wild.
# We just check that such models can be converted without error.
class TestFromMsAdditionalExamples:
    def test_durvasula_a_statistical_model(self):
        # Durvasula & Sankararaman (2019), https://doi.org/10.1371/journal.pgen.1008175
        # Similar use of simultaneous -es/-ej can be found in at least:
        # * Martin, Davey & Jiggins (2015), https://doi.org/10.1093/molbev/msu269
        # * Hibbins & Hahn (2019), https://doi.org/10.1534/genetics.118.301831
        # * Forsythe, Sloan & Beilstein (2020), https://doi.org/10.1093/gbe/evaa053
        demes.from_ms(
            # This example from the ArchIE git repository (Durvasula & Sankararaman).
            # https://github.com/sriramlab/ArchIE/blob/master/msmodified/readme
            """
            ms 240 1 -t 1000 -r 95.9652180334997 1000000 \
            -I 4 120 118 1 1 -en 0.06 1 0.01 -en 0.0605 1 1 \
            -es 0.0475 1 0.97 -ej 0.0475 5 3 -ej 0.0625 2 1 \
            -en 0.075 3 1 -en 0.078 3 1 -ej 0.225 4 3  -ej 0.325 3 1
            """,
            N0=N_ref(theta=1000, mu=1.25e-8, length=1000000),
        )

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_li_and_durbin(self):
        # Li & Durbin (2011), https://doi.org/10.1038/nature10231
        mu = 2.5e-8

        # Supplement page 1.
        demes.from_ms(
            "ms 2 100 -t 81960 -r 13560 30000000 "
            "-eN 0.01 0.05 -eN 0.0375 0.5 -eN 1.25 1",
            N0=N_ref(theta=81960, mu=mu, length=30000000),
        )
        demes.from_ms(
            "ms 2 100 -t 30000 -r 6000 30000000 "
            "-eN 0.01 0.1 -eN 0.06 1 -eN 0.2 0.5 -eN 1 1 -eN 2 2",
            N0=N_ref(theta=30000, mu=mu, length=30000000),
        )
        demes.from_ms(
            "ms 2 100 -t 3000 -r 600 30000000 "
            "-eN 0.1 5 -eN 0.6 20 -eN 2 5 -eN 10 10 -eN 20 5",
            N0=N_ref(theta=3000, mu=mu, length=30000000),
        )
        demes.from_ms(
            "ms 2 100 -t 60000 -r 12000 30000000 "
            "-eN 0.01 0.05 -eN 0.0150 0.5 -eN 0.05 0.25 -eN 0.5 0.5",
            N0=N_ref(theta=60000, mu=mu, length=30000000),
        )
        demes.from_ms(
            """
            ms 2 100 -t 65130.39 -r 10973.82 30000000 \
            -eN 0.0055 0.0832 -eN 0.0089 0.0489 \
            -eN 0.0130 0.0607 -eN 0.0177 0.1072 -eN 0.0233 0.2093 -eN 0.0299 0.3630 \
            -eN 0.0375 0.5041 -eN 0.0465 0.5870 -eN 0.0571 0.6343 -eN 0.0695 0.6138 \
            -eN 0.0840 0.5292 -eN 0.1010 0.4409 -eN 0.1210 0.3749 -eN 0.1444 0.3313 \
            -eN 0.1718 0.3066 -eN 0.2040 0.2952 -eN 0.2418 0.2915 -eN 0.2860 0.2950 \
            -eN 0.3379 0.3103 -eN 0.3988 0.3458 -eN 0.4701 0.4109 -eN 0.5538 0.5048 \
            -eN 0.6520 0.5996 -eN 0.7671 0.6440 -eN 0.9020 0.6178 -eN 1.0603 0.5345 \
            -eN 1.4635 1.7931
            """,
            N0=N_ref(theta=65130.39, mu=mu, length=30000000),
        )

        # XXX: misspecified!
        # The so-called "sim-split" model here ends with "-n 0.06 1",
        # which is a mistake because the first option arg must be an integer.
        # Paraphrasing, ms does the following (ms.c line 610 onwards):
        #          pop_idx = atoi(0.06) - 1;  // equals -1
        #          some_array[pop_idx] = 1;   // invalid write!
        # The resulting behaviour is thus undefined.
        #
        # demes.from_ms(
        #    "ms 2 100 -t 10000 -r 2000 10000000 "
        #    "-I 2 1 1 -n 1 1 -n 2 1 -ej 0.06 2 1 -n 0.06 1",
        #    N0=N_ref(theta=10000, mu=mu, length=10000000),
        # )

        # The lowercase '-l' in this command is not accepted by ms, and likely
        # should be uppercase '-L'.
        demes.from_ms(
            """
            ms 2 100 -t 10000 -r 2000 10000000 -T -l \
                -I 2 1 1 -eM 0 4 -eN 0 1 -en 0.01 1 0.1 \
                -eM 0.06 0 -ej 0.06 2 1 -eN 0.06 1 -eN 0.2 0.5 -eN 1 1 -eN 2 2
            """,
            N0=N_ref(theta=10000, mu=mu, length=10000000),
        )

        # Supplement page 5.
        demes.from_ms(
            """
            ms 2 100 -t 104693 -r 13862 30000000 -T -eN 0.0052 0.2504 \
            -eN 0.0084 0.1751 -es 0.0172 1 0.5 -en 0.0172 1 0.08755 \
            -en 0.0172 2 0.08755 -ej 0.0716 2 1 -eN 0.0716 0.1833 \
            -eN 0.1922 0.1885 -eN 0.2277 0.2022 -eN 0.2694 0.2295 \
            -eN 0.3183 0.2754 -eN 0.3756 0.3367 -eN 0.4428 0.3939 \
            -eN 0.5216 0.4190 -eN 0.6141 0.4104 -eN 0.7225 0.3954 \
            -eN 0.8496 0.3998 -eN 0.9987 0.5144 -eN 1.3785 1.8311
            """,
            N0=N_ref(theta=104693, mu=mu, length=30000000),
        )
        demes.from_ms(
            """
            ms 2 100 -t 104693 -r 13862 30000000 -T -eN 0.0052 0.2504 \
            -eN 0.0084 0.1751 -es 0.0172 1 0.33333 -es 0.0172 2 0.5 \
            -en 0.0172 1 0.08755 -en 0.0172 2 0.08755 \
            -ej 0.0716 3 2 -ej 0.0716 2 1 \
            -eN 0.0716 0.1833 \
            -eN 0.1922 0.1885 -eN 0.2277 0.2022 -eN 0.2694 0.2295 \
            -eN 0.3183 0.2754 -eN 0.3756 0.3367 -eN 0.4428 0.3939 \
            -eN 0.5216 0.4190 -eN 0.6141 0.4104 -eN 0.7225 0.3954 \
            -eN 0.8496 0.3998 -eN 0.9987 0.5144 -eN 1.3785 1.8311
            """,
            N0=N_ref(theta=104693, mu=mu, length=30000000),
        )

    def test_schiffels_inferring_human_population_size(self):
        # Schiffels & Durbin (2014), https://doi.org/10.1038/ng.3015
        mu = 1.25e-8
        # Split simulation
        demes.from_ms(
            """
            macs 4 30000000 -t 0.0007156 -r 0.0002 -I 2 2 2 -ej 0.116 2 1 -T
            """,
            N0=N_ref_macs(theta=0.0007156, mu=mu),
        )
        # Simulations with sharp population size changes (1).
        demes.from_ms(
            """
            macs 4 30000000 -t 0.0007156 -r 0.0002 -eN 0.0 10.8300726663 \
            -eN 0.00116452394261 1.08300726663 -eN 0.0174678591392 0.216601453326 \
            -eN 0.0465809577045 1.08300726663 -eN 0.0873392956959 3.24902179989 \
            -eN 0.232904788522 1.08300726663 -T
            """,
            N0=N_ref_macs(theta=0.0007156, mu=mu),
        )
        # Simulations with sharp population size changes (2).
        demes.from_ms(
            """
            macs 4 30000000 -t 0.001 -r 0.0004 -eN 0.0 8.25 -eN 0.0025 0.825 \
            -eN 0.0416666666667 2.475 -eN 0.166666666667 0.825 -T
            """,
            N0=N_ref_macs(theta=0.001, mu=mu),
        )
        # population split with migration
        demes.from_ms(
            """
            ms 4 10000000 -t 10000 -r 4000 -I 2 2 2 -ej 0.116 2 1 -eM 0.058 16 -T
            """,
            N0=N_ref(theta=10000, mu=mu, length=10000000),
        )
        # population split with population size changes
        demes.from_ms(
            """
            macs 4 30000000 -t 0.001 -r 0.0004 -I 2 2 2 -eN 0 9.5 \
            -en 0.000833333333333 1 0.95 -en 0.0025 2 0.95 -en 0.0125 1 0.19 \
            -en 0.0333333333333 1 0.95 -en 0.0416666666667 2 2.85 \
            -ej 0.05 2 1 -eN 0.166666666667 0.95 -T
            """,
            N0=N_ref_macs(theta=0.001, mu=mu),
        )

    def test_vernot_excavating_neandertal_and_denisovan_dna(self):
        # Vernot et al (2016), https://doi.org/10.1126/science.aad9416
        mu = 1.25e-8

        # modified Tennessen model.
        demes.from_ms(
            """
            macs 2025 15000000 -i 10 -r 3.0e-04 -t 0.00069 -T \
            -I 4 10 1006 1008 1 0 -n 4 0.205 -n 1 58.00274 -n 2 70.041 \
            -n 3 187.55 -eg 0.9e-10 1 482.46 -eg 1.0e-10 2 570.18 \
            -eg 1.1e-10 3 720.23 -em 1.2e-10 1 2 0.731 -em 1.3e-10 2 1 0.731 \
            -em 1.4e-10 3 1 0.2281 -em 1.5e-10 1 3 0.2281 \
            -em 1.6e-10 2 3 0.9094 -em 1.7e-10 3 2 0.9094 \
            -eg 0.007 1 0 -en 0.007001 1 1.98 -eg 0.007002 2 89.7668 \
            -eg 0.007003 3 113.3896 -eG 0.031456 0 -en 0.031457 2 0.1412 \
            -en 0.031458 3 0.07579 -eM 0.031459 0 -ej 0.03146 3 2 \
            -en 0.0314601 2 0.2546 -em 0.0314602 2 1 4.386 \
            -em 0.0314603 1 2 4.386 -eM 0.0697669 0 -ej 0.069767 2 1 \
            -en 0.0697671 1 1.98 -en 0.2025 1 1 -ej 0.9575923 4 1 \
            -em 0.06765 2 4 32 -em 0.06840 2 4 0
            """,
            N0=N_ref_macs(theta=0.00069, mu=mu),
        )
        # modified Gravel model (1).
        # XXX: misspecified! Possibly just an error when transcribing into
        # the manuscript. It shows a migration change:
        #    -em 0.314603 1 2 4.386
        # but it is clear this should be:
        #    -em 0.0314603 1 2 4.386
        # This is fixed below (and was correct in the second Gravel model)
        demes.from_ms(
            """
            macs 2025 15000000 -i 10 -r 3.0e-04 -t 0.00069 -T \
            -I 4 10 1006 1008 1 0 -n 4 0.205 -n 1 2.12 -n 2 4.911 -n 3 6.703 \
            -eg 1.0e-10 2 111.11 -eg 1.1e-10 3 140.35 -em 1.2e-10 1 2 0.731 \
            -em 1.3e-10 2 1 0.731 -em 1.4e-10 3 1 0.228 -em 1.5e-10 1 3 0.228 \
            -em 1.6e-10 2 3 0.9094 -em 1.7e-10 3 2 0.9094 -eG 0.031456 0 \
            -en 0.031457 2 0.1412 -en 0.031458 3 0.07579 -eM 0.031459 0 \
            -ej 0.03146 3 2 -en 0.0314601 2 0.2546 -em 0.0314602 2 1 4.386 \
            -em 0.0314603 1 2 4.386 -eM 0.0697669 0 -ej 0.069767 2 1 \
            -en 0.0697671 1 1.98 -en 0.2025 1 1 -ej 0.9575923 4 1 \
            -em 0.06765 2 4 32 -em 0.06840 2 4 0
            """,
            N0=N_ref_macs(theta=0.00069, mu=mu),
        )
        # modified Gravel model (2).
        demes.from_ms(
            """
            macs 2025 15000000 -i 10 -r 3.0e-04 -t 0.00069 -T \
            -I 4 10 1006 1008 1 0 -n 4 0.205 -n 1 2.12 -n 2 4.911 -n 3 6.703 \
            -eg 1.0e-10 2 78.95 -eg 1.1e-10 3 90.64 -em 1.2e-10 1 2 0.491 \
            -em 1.3e-10 2 1 0.491 -em 1.4e-10 3 1 0.1696 -em 1.5e-10 1 3 0.1696 \
            -em 1.6e-10 2 3 1.725 -em 1.7e-10 3 2 1.725 -eG 0.03826 0 \
            -en 0.03827 2 0.2216 -en 0.03828 3 0.1123 -eM 0.03829 0 \
            -ej 0.03830 3 2 -en 0.03831 2 0.3773 -em 0.03832 2 1 5.848 \
            -em 0.03833 1 2 5.848 -eM 0.1340 0 -ej 0.1341 2 1 -en 0.1342 1 2.105 \
            -en 0.4322 1 1 -ej 0.9575923 4 1 -em 0.06765 2 4 32 -em 0.06840 2 4 0
            """,
            N0=N_ref_macs(theta=0.00069, mu=mu),
        )
        # modified Gutenkunst model.
        demes.from_ms(
            """
            macs 2025 15000000 -i 10 -r 3.0e-04 -t 0.00069 -T \
            -I 4 10 1006 1008 1 0 -n 4 0.205 -n 1 1.685 -n 2 4.4 -n 3 8.6 \
            -eg 1.0e-10 2 116.8 -eg 1.1e-10 3 160.6 -em 1.2e-10 1 2 0.876 \
            -em 1.3e-10 2 1 0.876 -em 1.4e-10 3 1 0.5548 -em 1.5e-10 1 3 0.5548 \
            -em 1.6e-10 2 3 2.8032 -em 1.7e-10 3 2 .8032 -eG 0.0290 0 \
            -en 0.02901 2 0.1370 -en 0.02902 3 0.06986 -eM 0.02903 0 \
            -ej 0.02904 3 2 -en 0.0290401 2 0.2877 -em 0.0290402 2 1 7.3 \
            -em 0.0290403 1 2 7.3 -eM 0.19149 0 -ej 0.1915 2 1 \
            -en 0.191501 1 1.685 -en 0.3014 1 1 -ej 0.9575923 4 1 \
            -em 0.06774 2 4 34 -em 0.06849 2 4 0
            """,
            N0=N_ref_macs(theta=0.00069, mu=mu),
        )
        # modified Schaffner model.
        demes.from_ms(
            """
            macs 2025 15000000 -i 10 -r 3.0e-04 -t 0.00075 -T \
            -I 4 10 1006 1008 1 0 -n 4 0.205 -n 1 8 -n 2 8 -n 3 8 \
            -em 1.2e-10 1 2 1.6 -em 1.3e-10 2 1 1.6 -em 1.4e-10 3 1 0.4 \
            -em 1.5e-10 1 3 0.4 -en 0.004 1 1.92 -en 0.007 2 0.616 \
            -en 0.008 3 0.616 -en 0.03942 2 0.0574 -en 0.03998 2 0.616 \
            -en 0.038 3 0.058 -en 0.03997 3 0.616  -eM 0.03999 0 -ej 0.040 3 2 \
            -en 0.04001 2 0.616 -en 0.0686 2 0.032 -en 0.0696 1 0.0996 \
            -ej 0.07 2 1 -en 0.07001 1 1.92 -en 0.34 1 1 -ej 0.56 4 1 \
            -em 0.04002 2 4 29 -em 0.04077 2 4 0
            """,
            N0=N_ref_macs(theta=0.00069, mu=mu),
        )

    def test_soraggi_powerful_inference_with_the_D_statistic(self):
        # Soraggi, Wuif & Albrechtsen (2018), https://doi.org/10.1534/g3.117.300192
        # msms commands use '-ms nsamples nreps', which is rejected by argparse
        # because we define the '-m' option which is a prefix of '-ms'.
        # The '-ms' options have been removed from the commands below.
        demes.from_ms(
            """
            msms -N 10000 -I 4 10 10 10 10 0 -t 100 -r 100 1000 \
            -em 0.2 3 1 16 -em 0.201 3 1 0 -ej 0.5 1 2 -ej 0.75 2 3 -ej 1 3 4
            """,
            N0=10000,
        )
        demes.from_ms(
            """
            msms -N 10000 -I 4 2 2 2 2 0 -t 100 -r 100 1000 \
            -ej 0.5 1 2 -ej 0.75 2 3 -ej 1 3 4
            """,
            N0=10000,
        )
        demes.from_ms(
            """
            msms -N 10000 -I 5 10 10 10 10 10 0 -t 100 -r 100 1000 \
            -es 0.1 1 0.9 -ej 0.2 6 5 -ej 0.25 1 2 -ej 0.5 2 3 -ej 0.75 3 4 \
            -ej 30 4 5
            """,
            N0=10000,
        )


class TestOptionStrings:
    def test_structure_str(self):
        assert str(ms.Structure.from_nargs(1, 2)) == "-I 1 2"
        assert str(ms.Structure.from_nargs(2, 2, 2)) == "-I 2 2 2"
        assert str(ms.Structure.from_nargs(3, 2, 2, 0, 0.1)) == "-I 3 2 2 0 0.1"

    def test_growth_rate_change_str(self):
        assert str(ms.GrowthRateChange(0, 1e-5)) == "-G 1e-05"
        assert str(ms.GrowthRateChange(0, -1e-5)) == "-G -0.0000100000"
        assert str(ms.GrowthRateChange(1.2, 1e-5)) == "-eG 1.2 1e-05"
        assert str(ms.GrowthRateChange(5, -1e-5)) == "-eG 5.0 -0.0000100000"

    def test_population_growth_rate_change_str(self):
        assert str(ms.PopulationGrowthRateChange(0, 1, 1e-5)) == "-g 1 1e-05"
        assert str(ms.PopulationGrowthRateChange(0, 2, -1e-5)) == "-g 2 -0.0000100000"
        assert str(ms.PopulationGrowthRateChange(1.2, 3, 1e-5)) == "-eg 1.2 3 1e-05"
        assert (
            str(ms.PopulationGrowthRateChange(5, 4, -1e-5)) == "-eg 5.0 4 -0.0000100000"
        )

    def test_size_change_str(self):
        assert str(ms.SizeChange(0, 2.5)) == "-eN 0.0 2.5"
        assert str(ms.SizeChange(0.5, 5.5)) == "-eN 0.5 5.5"
        assert str(ms.SizeChange(1, 10)) == "-eN 1.0 10.0"

    def test_population_size_change_str(self):
        assert str(ms.PopulationSizeChange(0, 1, 2.5)) == "-n 1 2.5"
        assert str(ms.PopulationSizeChange(0, 10, 25)) == "-n 10 25.0"
        assert str(ms.PopulationSizeChange(0.5, 2, 5.5)) == "-en 0.5 2 5.5"
        assert str(ms.PopulationSizeChange(1, 3, 10)) == "-en 1.0 3 10.0"

    def test_migration_rate_change_str(self):
        assert str(ms.MigrationRateChange(0, 1)) == "-eM 0.0 1.0"
        assert str(ms.MigrationRateChange(1, 2)) == "-eM 1.0 2.0"
        assert str(ms.MigrationRateChange(1.5, 1e-5)) == "-eM 1.5 1e-05"

    def test_migration_matrix_entry_change_str(self):
        assert str(ms.MigrationMatrixEntryChange(0, 1, 2, 1e-5)) == "-m 1 2 1e-05"
        assert str(ms.MigrationMatrixEntryChange(0, 5, 3, 15)) == "-m 5 3 15.0"
        assert str(ms.MigrationMatrixEntryChange(1, 1, 2, 1e-5)) == "-em 1.0 1 2 1e-05"
        assert (
            str(ms.MigrationMatrixEntryChange(0.123, 5, 3, 15)) == "-em 0.123 5 3 15.0"
        )

    def test_migration_matrix_change_str(self):
        assert (
            str(ms.MigrationMatrixChange.from_nargs(0, 2, -100, 1e-5, 1e-6, -100))
            == "-ma 2 x 1e-05 1e-06 x"
        )
        assert (
            str(
                ms.MigrationMatrixChange.from_nargs(
                    0, 3, -100, 1e-5, 1e-6, 1e-7, -100, 1e-8, 1e-9, 1e-10, -100
                )
            )
            == "-ma 3 x 1e-05 1e-06 1e-07 x 1e-08 1e-09 1e-10 x"
        )
        assert (
            str(ms.MigrationMatrixChange.from_nargs(1, 2, -100, 1e-5, 1e-6, -100))
            == "-ema 1.0 2 x 1e-05 1e-06 x"
        )
        assert (
            str(
                ms.MigrationMatrixChange.from_nargs(
                    2.5, 3, -100, 1e-5, 1e-6, 1e-7, -100, 1e-8, 1e-9, 1e-10, -100
                )
            )
            == "-ema 2.5 3 x 1e-05 1e-06 1e-07 x 1e-08 1e-09 1e-10 x"
        )

    def test_split_str(self):
        assert str(ms.Split(0, 1, 0.2)) == "-es 0.0 1 0.2"
        assert str(ms.Split(0.1, 6, 1e-5)) == "-es 0.1 6 1e-05"
        assert str(ms.Split(1, 632, 1 / 3)) == "-es 1.0 632 0.3333333333333333"

    def test_join_str(self):
        assert str(ms.Join(0, 1, 2)) == "-ej 0.0 1 2"
        assert str(ms.Join(0.1, 6, 5)) == "-ej 0.1 6 5"
        assert str(ms.Join(1, 632, 1)) == "-ej 1.0 632 1"


class TestToMs:
    def parse_command(self, cmd):
        parser = ms.build_parser()
        args = parser.parse_args(cmd.split())
        events = args.initial_state + args.demographic_events
        return args.structure, events

    def test_one_deme_constant_size(self):
        N0 = 100
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        graph = b.resolve()
        cmd = demes.to_ms(graph, N0=N0)
        # Nothing to be done.
        assert cmd == ""

        # Set N0 differently to deme size.
        cmd = demes.to_ms(graph, N0=N0 / 50)
        structure, events = self.parse_command(cmd)
        assert structure is None
        assert len(events) == 1
        assert isinstance(events[0], (ms.SizeChange, ms.PopulationSizeChange))
        assert math.isclose(events[0].t, 0)
        assert math.isclose(events[0].x, 50)

    def test_one_deme_piecewise_constant(self):
        N0, N1, N2 = 100, 200, 3000
        T0, T1, T2 = 0, 500, 600
        b = demes.Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=N2, end_time=T2),
                dict(start_size=N1, end_time=T1),
                dict(start_size=N0, end_time=T0),
            ],
        )
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure is None
        assert len(events) == 2
        assert isinstance(events[0], (ms.SizeChange, ms.PopulationSizeChange))
        assert math.isclose(events[0].t, T1 / (4 * N0))
        assert math.isclose(events[0].x, N1 / N0)
        assert isinstance(events[1], (ms.SizeChange, ms.PopulationSizeChange))
        assert math.isclose(events[1].t, T2 / (4 * N0))
        assert math.isclose(events[1].x, N2 / N0)

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    def test_one_deme_piecewise_nonconstant(self):
        N0, N1, N2 = 100, 200, 3000
        T0, T1, T2 = 0, 500, 600
        b = demes.Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=N2, end_time=T2),
                dict(start_size=N2, end_size=N1, end_time=T1),
                dict(start_size=N1, end_size=N0, end_time=T0),
            ],
        )
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure is None
        assert len(events) == 3
        assert isinstance(
            events[0], (ms.GrowthRateChange, ms.PopulationGrowthRateChange)
        )
        assert events[0].t == 0
        assert math.isclose(events[0].alpha, 4 * N0 * -math.log(N1 / N0) / (T1 - T0))
        assert isinstance(
            events[1], (ms.GrowthRateChange, ms.PopulationGrowthRateChange)
        )
        assert math.isclose(events[1].t, T1 / (4 * N0))
        assert math.isclose(events[1].alpha, 4 * N0 * -math.log(N2 / N1) / (T2 - T1))
        assert isinstance(
            events[2], (ms.GrowthRateChange, ms.PopulationGrowthRateChange)
        )
        assert math.isclose(events[2].t, T2 / (4 * N0))
        assert math.isclose(events[2].alpha, 0)

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    @pytest.mark.parametrize("num_demes", [2, 5, 10])
    def test_multiple_demes_constant_size(self, num_demes):
        N0 = 100
        b = demes.Builder()
        for j in range(num_demes):
            b.add_deme(f"deme{j}", epochs=[dict(start_size=N0)])
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == num_demes
        assert structure.rate == 0
        assert len(events) == 0

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    @pytest.mark.parametrize("num_demes", [2, 5, 10])
    def test_multiple_demes_with_size_change(self, num_demes):
        N0, N1 = 100, 200
        T0, T1 = 0, 500
        b = demes.Builder()
        for j in range(num_demes - 1):
            b.add_deme(f"deme{j}", epochs=[dict(start_size=N0)])
        b.add_deme(
            "x",
            epochs=[
                dict(start_size=N1, end_time=T1),
                dict(start_size=N0, end_time=T0),
            ],
        )
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == num_demes
        assert structure.rate == 0
        assert len(events) == 1
        assert isinstance(events[0], ms.PopulationSizeChange)
        assert events[0].i == num_demes
        assert math.isclose(events[0].t, T1 / (4 * N0))
        assert math.isclose(events[0].x, N1 / N0)

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    @pytest.mark.parametrize("num_demes", [2, 5, 10])
    def test_multiple_demes_with_growth_rate_change(self, num_demes):
        N0, N1 = 100, 200
        T0, T1 = 0, 500
        b = demes.Builder()
        for j in range(num_demes - 1):
            b.add_deme(f"deme{j}", epochs=[dict(start_size=N0)])
        b.add_deme(
            "x",
            epochs=[
                dict(start_size=N1, end_size=N1, end_time=T1),
                dict(start_size=N1, end_size=N0, end_time=T0),
            ],
        )
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == num_demes
        assert structure.rate == 0
        assert len(events) == 2
        assert isinstance(events[0], ms.PopulationGrowthRateChange)
        assert events[0].i == num_demes
        assert events[0].t == 0
        assert math.isclose(events[0].alpha, 4 * N0 * -math.log(N1 / N0) / (T1 - T0))
        assert isinstance(events[1], ms.PopulationGrowthRateChange)
        assert events[1].i == num_demes
        assert math.isclose(events[1].t, T1 / (4 * N0))
        assert math.isclose(events[1].alpha, 0)

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    def test_single_ancestor(self):
        N0 = 100
        T0 = 50
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", start_time=T0, ancestors=["a"], epochs=[dict(start_size=N0)])
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == 2
        assert structure.rate == 0
        assert len(events) == 1
        assert isinstance(events[0], ms.Join)
        assert events[0].i == 2
        assert events[0].j == 1
        assert math.isclose(events[0].t, T0 / (4 * N0))

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        assert graph2["a"].ancestors == []
        assert graph2["b"].ancestors == ["a"]
        assert math.isclose(graph2["b"].start_time, T0)

    def test_multiple_ancestors(self):
        N0 = 100
        T0 = 50
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        b.add_deme(
            "c",
            start_time=T0,
            ancestors=["a", "b"],
            proportions=[0.1, 0.9],
            epochs=[dict(start_size=N0)],
        )
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)

        # There are multiple ways this model could be turned into ms commands,
        # so we just check that the ancestry matches after converting from_ms
        # back to a demes graph.
        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        assert graph2["a"].ancestors == []
        assert graph2["b"].ancestors == []
        ancestors, proportions = zip(
            *sorted(zip(graph2["c"].ancestors, graph2["c"].proportions))
        )
        assert ancestors == ("a", "b")
        assert math.isclose(proportions[0], 0.1)
        assert math.isclose(proportions[1], 0.9)
        assert math.isclose(graph2["c"].start_time, T0)

    def test_pulse(self):
        N0 = 100
        T0 = 50
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        b.add_pulse(sources=["a"], dest="b", time=T0, proportions=[0.1])
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == 2
        assert structure.rate == 0
        assert len(events) == 2
        assert isinstance(events[0], ms.Split)
        assert events[0].i == 2
        assert math.isclose(events[0].p, 1 - 0.1)
        assert math.isclose(events[0].t, T0 / (4 * N0))
        assert isinstance(events[1], ms.Join)
        assert events[1].i == 3
        assert events[1].j == 1
        assert math.isclose(events[1].t, T0 / (4 * N0))

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    @pytest.mark.filterwarnings("ignore:Multiple pulses.*same.*time")
    def test_pulse_order(self):
        N0 = 100
        T0 = 50
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        b.add_deme("c", epochs=[dict(start_size=N0)])
        b.add_pulse(sources=["a"], dest="b", time=T0, proportions=[0.1])
        b.add_pulse(sources=["b"], dest="c", time=T0, proportions=[0.1])
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)

        structure, events = self.parse_command(cmd)
        assert structure.npop == 3
        assert structure.rate == 0
        assert len(events) == 4
        assert isinstance(events[0], ms.Split)
        assert isinstance(events[1], ms.Join)
        assert isinstance(events[2], ms.Split)
        assert isinstance(events[3], ms.Join)

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

        # Swap order of pulses, and confirm the output order gets swapped.
        b.data["pulses"] = [b.data["pulses"][1], b.data["pulses"][0]]
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    @pytest.mark.parametrize("num_demes", [2, 5, 10])
    def test_isolation_with_migration(self, num_demes):
        N0 = 100
        mig_rate = 1e-3
        b = demes.Builder()
        deme_names = [f"d{j}" for j in range(num_demes)]
        for j in range(num_demes):
            b.add_deme(deme_names[j], epochs=[dict(start_size=N0)])
        b.add_migration(demes=deme_names, rate=mig_rate)
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == num_demes
        # One possible implementation:
        # assert structure.rate == mig_rate / (4 * N0) / (num_demes - 1)

        # There are many possible ways to specify migrations, and it's not so
        # important which is implemented. So just check the from_ms round trip.
        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    def test_migration_asymmetric(self):
        N0 = 100
        T0, T1 = 100, 200
        mig_rate = 1e-3
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        b.add_migration(source="a", dest="b", rate=mig_rate, start_time=T1, end_time=T0)
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == 2
        assert structure.rate == 0
        assert len(events) == 2
        # This is tied to the implementation, which outputs '-em' events.
        assert isinstance(events[0], (ms.MigrationMatrixEntryChange))
        assert math.isclose(events[0].t, T0 / (4 * N0))
        assert math.isclose(events[0].rate, mig_rate * 4 * N0)
        assert events[0].i == 2
        assert events[0].j == 1
        assert isinstance(events[1], (ms.MigrationMatrixEntryChange))
        assert math.isclose(events[1].t, T1 / (4 * N0))
        assert math.isclose(events[1].rate, 0)
        assert events[1].i == 2
        assert events[1].j == 1

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    def test_migrations_overlapping(self):
        N0 = 100
        T0a, T0b = 100, 200
        T1a, T1b = 150, 250
        M0, M1 = 1e-3, 1e-4
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        b.add_deme("c", epochs=[dict(start_size=N0)])
        b.add_migration(source="a", dest="b", rate=M0, start_time=T0b, end_time=T0a)
        b.add_migration(demes=["b", "c"], rate=M1, start_time=T1b, end_time=T1a)
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == 3
        assert structure.rate == 0
        assert len(events) == (1 + 2) * 2

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    def test_migrations_consecutive(self):
        N0 = 100
        T0, T1, T2 = 100, 200, 300
        M0, M1 = 1e-3, 1e-4
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        b.add_migration(source="a", dest="b", rate=M0, start_time=T1, end_time=T0)
        b.add_migration(demes=["a", "b"], rate=M1, start_time=T2, end_time=T1)
        graph1 = b.resolve()
        cmd = demes.to_ms(graph1, N0=N0)
        structure, events = self.parse_command(cmd)
        assert structure.npop == 2
        assert structure.rate == 0
        assert len(events) == (1 + 2) * 2

        graph2 = demes.from_ms(
            cmd, N0=N0, deme_names=[deme.name for deme in graph1.demes]
        )
        graph2.assert_close(graph1)

    def test_size_function_unsupported(self):
        N0, N1 = 100, 200
        T0 = 100
        b = demes.Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=N0, end_time=T0),
                dict(start_size=N0, end_size=N1, size_function="linear"),
            ],
        )
        graph1 = b.resolve()
        with pytest.raises(
            ValueError, match="ms only supports constant or exponential"
        ):
            demes.to_ms(graph1, N0=N0)

    def test_sample_configuration(self):
        N0 = 100
        b = demes.Builder()
        b.add_deme("a", epochs=[dict(start_size=N0)])
        b.add_deme("b", epochs=[dict(start_size=N0)])
        graph = b.resolve()
        demes.to_ms(graph, N0=N0, samples=[2, 0])
        demes.to_ms(graph, N0=N0, samples=[1, 1])
        demes.to_ms(graph, N0=N0, samples=[0, 2])

        for bad_samples in ([], [2], [2, 2, 2]):
            with pytest.raises(
                ValueError, match="samples must match the number of demes"
            ):
                demes.to_ms(graph, N0=N0, samples=bad_samples)

    def test_multisource_pulse(self):
        b = demes.Builder(defaults=dict(epoch=dict(start_size=100, end_time=0)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme("c")
        b.add_pulse(sources=["a", "b"], dest="c", time=10, proportions=[0.2, 0.2])
        graph = b.resolve()
        with pytest.raises(
            ValueError, match="pulses with only a single source are supported"
        ):
            demes.to_ms(graph, N0=100)
