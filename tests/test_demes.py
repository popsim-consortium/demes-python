import unittest
import copy
import pathlib
import math
import typing

import pytest
import hypothesis as hyp

from demes import (
    Builder,
    Epoch,
    AsymmetricMigration,
    SymmetricMigration,
    Pulse,
    Deme,
    Graph,
    Split,
    Branch,
    Merge,
    Admix,
    load,
)
import demes
import tests


class TestEpoch(unittest.TestCase):
    def test_bad_time(self):
        for time in ("0", "inf", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=time,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=100,
                    end_time=time,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )

        for start_time in (-10000, -1, -1e-9):
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=start_time,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )
        for end_time in (-10000, -1, -1e-9, math.inf):
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=100,
                    end_time=end_time,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )

    def test_bad_time_span(self):
        with self.assertRaises(ValueError):
            Epoch(
                start_time=1,
                end_time=1,
                start_size=1,
                end_size=1,
                size_function="constant",
            )
        with self.assertRaises(ValueError):
            Epoch(
                start_time=1,
                end_time=2,
                start_size=1,
                end_size=1,
                size_function="constant",
            )

    def test_bad_size(self):
        for size in ("0", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=1,
                    end_time=0,
                    start_size=size,
                    end_size=1,
                    size_function="exponential",
                )
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=1,
                    end_time=0,
                    start_size=1,
                    end_size=size,
                    size_function="exponential",
                )

        for size in (-10000, -1, -1e-9, 0, math.inf):
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=1,
                    end_time=0,
                    start_size=size,
                    end_size=1,
                    size_function="exponential",
                )
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=1,
                    end_time=0,
                    start_size=1,
                    end_size=size,
                    size_function="exponential",
                )

    def test_missing_values(self):
        with self.assertRaises(TypeError):
            Epoch()
        with self.assertRaises(TypeError):
            Epoch(start_time=1, end_time=0, start_size=1, end_size=1)
        with self.assertRaises(TypeError):
            Epoch(start_time=1, end_time=0, start_size=1, size_function="constant")
        with self.assertRaises(TypeError):
            Epoch(start_time=1, end_time=0, end_size=1, size_function="constant")
        with self.assertRaises(TypeError):
            Epoch(start_time=1, start_size=1, end_size=1, size_function="constant")
        with self.assertRaises(TypeError):
            Epoch(end_time=0, start_size=1, end_size=1, size_function="constant")

    def test_valid_epochs(self):
        Epoch(
            start_time=math.inf,
            end_time=0,
            start_size=1,
            end_size=1,
            size_function="constant",
        )
        Epoch(
            start_time=math.inf,
            end_time=10,
            start_size=1,
            end_size=1,
            size_function="constant",
        )
        Epoch(
            start_time=100,
            end_time=99,
            start_size=1,
            end_size=1,
            size_function="constant",
        )
        Epoch(
            start_time=100,
            end_time=0,
            start_size=1,
            end_size=100,
            size_function="exponential",
        )
        Epoch(
            start_time=20,
            end_time=10,
            start_size=1,
            end_size=100,
            size_function="exponential",
        )
        for rate in (0, 1, 0.5, 1e-5):
            Epoch(
                start_time=math.inf,
                end_time=0,
                start_size=1,
                end_size=1,
                size_function="constant",
                selfing_rate=rate,
            )
            Epoch(
                start_time=math.inf,
                end_time=0,
                start_size=1,
                end_size=1,
                size_function="constant",
                cloning_rate=rate,
            )
            Epoch(
                start_time=math.inf,
                end_time=0,
                start_size=1,
                end_size=1,
                size_function="constant",
                selfing_rate=rate,
                cloning_rate=1 - rate,
            )
        for fn in ("linear", "exponential", "N(t) = 6 * log(t)"):
            Epoch(
                start_time=100, end_time=0, start_size=1, end_size=10, size_function=fn
            )

    def test_time_span(self):
        e = Epoch(
            start_time=math.inf,
            end_time=0,
            start_size=1,
            end_size=1,
            size_function="constant",
        )
        self.assertEqual(e.time_span, math.inf)
        e = Epoch(
            start_time=100,
            end_time=20,
            start_size=1,
            end_size=1,
            size_function="constant",
        )
        self.assertEqual(e.time_span, 80)

    def test_inf_start_time_constant_epoch(self):
        with self.assertRaises(ValueError):
            Epoch(
                start_time=math.inf,
                end_time=0,
                start_size=10,
                end_size=20,
                size_function="exponential",
            )

    def test_isclose(self):
        eps = 1e-50
        e1 = Epoch(
            start_time=10,
            end_time=0,
            start_size=1,
            end_size=1,
            size_function="exponential",
        )
        self.assertTrue(e1.isclose(e1))
        self.assertTrue(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=0 + eps,
                    start_size=1,
                    end_size=1,
                    size_function="exponential",
                )
            )
        )
        self.assertTrue(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1 + eps,
                    end_size=1,
                    size_function="exponential",
                )
            )
        )

        self.assertFalse(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=1e-9,
                    start_size=1,
                    end_size=1,
                    size_function="exponential",
                )
            )
        )
        self.assertFalse(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1 + 1e-9,
                    end_size=1,
                    size_function="exponential",
                )
            )
        )
        self.assertFalse(
            e1.isclose(
                Epoch(
                    start_time=math.inf,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="exponential",
                )
            )
        )
        self.assertFalse(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )
            )
        )
        self.assertFalse(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                    selfing_rate=0.1,
                )
            )
        )
        self.assertFalse(
            e1.isclose(
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                    cloning_rate=0.1,
                )
            )
        )
        self.assertFalse(e1.isclose(None))
        self.assertFalse(e1.isclose(123))
        self.assertFalse(e1.isclose("foo"))

    def test_bad_selfing_rate(self):
        for rate in ("0", "1e-4", "inf", [], {}, math.nan):
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=100,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                    selfing_rate=rate,
                )

        for rate in (-10000, 10000, -1, -1e-9, 1.2, math.inf):
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=100,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                    selfing_rate=rate,
                )

    def test_bad_cloning_rate(self):
        for rate in ("0", "1e-4", "inf", [], {}, math.nan):
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=100,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                    cloning_rate=rate,
                )

        for rate in (-10000, 10000, -1, -1e-9, 1.2, math.inf):
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=100,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                    cloning_rate=rate,
                )

    def test_bad_selfing_rate_cloning_rate_combination(self):
        with self.assertRaises(ValueError):
            Epoch(
                start_time=100,
                end_time=0,
                start_size=1,
                end_size=1,
                size_function="constant",
                cloning_rate=0.5,
                selfing_rate=0.5 + 1e-9,
            )
        with self.assertRaises(ValueError):
            Epoch(
                start_time=100,
                end_time=0,
                start_size=1,
                end_size=1,
                size_function="constant",
                cloning_rate=1,
                selfing_rate=1,
            )

    def test_bad_size_function(self):
        for fn in (0, 1e5, [], {}, math.nan):
            with self.assertRaises(TypeError):
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1,
                    end_size=10,
                    size_function=fn,
                )

        for fn in ("", "constant"):
            with self.assertRaises(ValueError):
                Epoch(
                    start_time=10,
                    end_time=0,
                    start_size=1,
                    end_size=10,
                    size_function=fn,
                )


class TestMigration(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                AsymmetricMigration(
                    source="a", dest="b", start_time=time, end_time=0, rate=0.1
                )
            with self.assertRaises(TypeError):
                AsymmetricMigration(
                    source="a", dest="b", start_time=100, end_time=time, rate=0.1
                )
            with self.assertRaises(TypeError):
                SymmetricMigration(
                    demes=["a", "b"], start_time=time, end_time=0, rate=0.1
                )
            with self.assertRaises(TypeError):
                SymmetricMigration(
                    demes=["a", "b"], start_time=100, end_time=time, rate=0.1
                )

        for time in (-10000, -1, -1e-9):
            with self.assertRaises(ValueError):
                AsymmetricMigration(
                    source="a", dest="b", start_time=time, end_time=0, rate=0.1
                )
            with self.assertRaises(ValueError):
                SymmetricMigration(
                    demes=["a", "b"], start_time=time, end_time=0, rate=0.1
                )
        for time in (-10000, -1, -1e-9, math.inf):
            with self.assertRaises(ValueError):
                AsymmetricMigration(
                    source="a", dest="b", start_time=100, end_time=time, rate=0.1
                )
            with self.assertRaises(ValueError):
                SymmetricMigration(
                    demes=["a", "b"], start_time=100, end_time=time, rate=0.1
                )

        # start_time == end_time
        with self.assertRaises(ValueError):
            AsymmetricMigration(
                source="a", dest="b", start_time=100, end_time=100, rate=0.1
            )
        with self.assertRaises(ValueError):
            SymmetricMigration(demes=["a", "b"], start_time=100, end_time=100, rate=0.1)

        # start_time < end_time
        with self.assertRaises(ValueError):
            AsymmetricMigration(
                source="a", dest="b", start_time=10, end_time=100, rate=0.1
            )
        with self.assertRaises(ValueError):
            SymmetricMigration(demes=["a", "b"], start_time=10, end_time=100, rate=0.1)

    def test_bad_rate(self):
        for rate in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                AsymmetricMigration(
                    source="a", dest="b", start_time=10, end_time=0, rate=rate
                )
            with self.assertRaises(TypeError):
                SymmetricMigration(
                    demes=["a", "b"], start_time=10, end_time=0, rate=rate
                )

        for rate in (-10000, -1, -1e-9, 1.2, 100, math.inf):
            with self.assertRaises(ValueError):
                AsymmetricMigration(
                    source="a", dest="b", start_time=10, end_time=0, rate=rate
                )
            with self.assertRaises(ValueError):
                SymmetricMigration(
                    demes=["a", "b"], start_time=10, end_time=0, rate=rate
                )

    def test_bad_demes(self):
        for name in (0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                AsymmetricMigration(
                    source=name, dest="a", start_time=10, end_time=0, rate=0.1
                )
            with self.assertRaises(TypeError):
                AsymmetricMigration(
                    source="a", dest=name, start_time=10, end_time=0, rate=0.1
                )
            with self.assertRaises(TypeError):
                SymmetricMigration(
                    demes=["a", name], start_time=10, end_time=0, rate=0.1
                )

        for name in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                AsymmetricMigration(
                    source=name, dest="a", start_time=10, end_time=0, rate=0.1
                )
            with self.assertRaises(ValueError):
                AsymmetricMigration(
                    source="a", dest=name, start_time=10, end_time=0, rate=0.1
                )
            with self.assertRaises(ValueError):
                SymmetricMigration(
                    demes=["a", name], start_time=10, end_time=0, rate=0.1
                )

        # need at least two demes for symmetric migration
        with self.assertRaises(ValueError):
            SymmetricMigration(demes=["a"], start_time=10, end_time=0, rate=0.1)
        with self.assertRaises(ValueError):
            SymmetricMigration(demes=[], start_time=10, end_time=0, rate=0.1)

    def test_valid_migration(self):
        AsymmetricMigration(
            source="a", dest="b", start_time=math.inf, end_time=0, rate=1e-9
        )
        AsymmetricMigration(
            source="a", dest="b", start_time=1000, end_time=999, rate=0.9
        )
        SymmetricMigration(demes=["a", "b"], start_time=math.inf, end_time=0, rate=1e-9)
        SymmetricMigration(demes=["a", "b"], start_time=1000, end_time=999, rate=0.9)
        SymmetricMigration(
            demes=["a", "b", "c", "d", "e"], start_time=1, end_time=0, rate=0.9
        )

    def test_isclose(self):
        eps = 1e-50
        m1 = AsymmetricMigration(
            source="a", dest="b", start_time=1, end_time=0, rate=1e-9
        )
        self.assertTrue(m1.isclose(m1))
        self.assertTrue(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="b", start_time=1, end_time=0, rate=1e-9 + eps
                )
            )
        )
        self.assertTrue(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="b", start_time=1 + eps, end_time=0, rate=1e-9
                )
            )
        )
        self.assertTrue(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="b", start_time=1, end_time=0 + eps, rate=1e-9
                )
            )
        )

        self.assertFalse(
            m1.isclose(
                AsymmetricMigration(
                    source="b", dest="a", start_time=1, end_time=0, rate=1e-9
                )
            )
        )
        self.assertFalse(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="b", start_time=1, end_time=0, rate=2e-9
                )
            )
        )
        self.assertFalse(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="c", start_time=1, end_time=0, rate=1e-9
                )
            )
        )
        self.assertFalse(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="c", start_time=2, end_time=0, rate=1e-9
                )
            )
        )
        self.assertFalse(
            m1.isclose(
                AsymmetricMigration(
                    source="a", dest="c", start_time=1, end_time=0.1, rate=1e-9
                )
            )
        )
        self.assertFalse(m1.isclose(None))
        self.assertFalse(m1.isclose(123))
        self.assertFalse(m1.isclose("foo"))


class TestPulse(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Pulse(source="a", dest="b", time=time, proportion=0.1)

        for time in (-10000, -1, -1e-9, 0, math.inf):
            with self.assertRaises(ValueError):
                Pulse(source="a", dest="b", time=time, proportion=0.1)

    def test_bad_proportion(self):
        for proportion in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Pulse(source="a", dest="b", time=1, proportion=proportion)

        for proportion in (-10000, -1, -1e-9, 1.2, 100, math.inf):
            with self.assertRaises(ValueError):
                Pulse(source="a", dest="b", time=1, proportion=proportion)

    def test_bad_demes(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Pulse(source=name, dest="a", time=1, proportion=0.1)
            with self.assertRaises(TypeError):
                Pulse(source="a", dest=name, time=1, proportion=0.1)

        for name in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Pulse(source=name, dest="a", time=1, proportion=0.1)
            with self.assertRaises(ValueError):
                Pulse(source="a", dest=name, time=1, proportion=0.1)

    def test_valid_pulse(self):
        Pulse(source="a", dest="b", time=1, proportion=1e-9)
        Pulse(source="a", dest="b", time=100, proportion=0.9)

    def test_isclose(self):
        eps = 1e-50
        p1 = Pulse(source="a", dest="b", time=1, proportion=1e-9)
        self.assertTrue(p1.isclose(p1))
        self.assertTrue(
            p1.isclose(Pulse(source="a", dest="b", time=1, proportion=1e-9))
        )
        self.assertTrue(
            p1.isclose(Pulse(source="a", dest="b", time=1 + eps, proportion=1e-9))
        )
        self.assertTrue(
            p1.isclose(Pulse(source="a", dest="b", time=1, proportion=1e-9 + eps))
        )

        self.assertFalse(
            p1.isclose(Pulse(source="a", dest="c", time=1, proportion=1e-9))
        )
        self.assertFalse(
            p1.isclose(Pulse(source="b", dest="a", time=1, proportion=1e-9))
        )
        self.assertFalse(
            p1.isclose(Pulse(source="a", dest="b", time=1, proportion=2e-9))
        )
        self.assertFalse(
            p1.isclose(Pulse(source="a", dest="b", time=1 + 1e-9, proportion=1e-9))
        )


class TestSplit(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Split(parent="a", children=["b", "c"], time=time)

        for time in [-1e-12, -1, math.inf]:
            with self.assertRaises(ValueError):
                Split(parent="a", children=["b", "c"], time=time)

    def test_bad_children(self):
        for children in (None, "b", {"b": 1}, set("b"), ("b",)):
            with self.assertRaises(TypeError):
                Split(parent="a", children=children, time=1)
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Split(parent="a", children=[name], time=1)

        for children in (["a", "b"], ["b", "b"], []):
            with self.assertRaises(ValueError):
                Split(parent="a", children=children, time=1)
        for name in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Split(parent="a", children=[name], time=1)

    def test_bad_parent(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Split(parent=name, children=["b"], time=1)

        for name in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Split(parent=name, children=["a"], time=1)

    def test_valid_split(self):
        Split(parent="a", children=["b", "c"], time=10)
        Split(parent="a", children=["b", "c", "d"], time=10)
        # TODO: a split at time=0 should probably be an error
        Split(parent="a", children=["b", "c"], time=0)

    def test_isclose(self):
        eps = 1e-50
        s1 = Split(parent="a", children=["b", "c"], time=1)
        self.assertTrue(s1.isclose(s1))
        self.assertTrue(s1.isclose(Split(parent="a", children=["b", "c"], time=1)))
        self.assertTrue(
            s1.isclose(Split(parent="a", children=["b", "c"], time=1 + eps))
        )
        # Order of children doesn't matter.
        self.assertTrue(s1.isclose(Split(parent="a", children=["c", "b"], time=1)))

        self.assertFalse(s1.isclose(Split(parent="a", children=["x", "c"], time=1)))
        self.assertFalse(s1.isclose(Split(parent="x", children=["b", "c"], time=1)))
        self.assertFalse(
            s1.isclose(Split(parent="a", children=["b", "c", "x"], time=1))
        )
        self.assertFalse(
            s1.isclose(Split(parent="a", children=["b", "c"], time=1 + 1e-9))
        )


class TestBranch(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Branch(parent="a", child="b", time=time)

        for time in [-1e-12, -1, math.inf]:
            with self.assertRaises(ValueError):
                Branch(parent="a", child="b", time=time)

    def test_bad_child(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Branch(parent="a", child=name, time=1)

        for name in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Branch(parent="a", child=name, time=1)

    def test_bad_parent(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Branch(parent=name, child="b", time=1)

        for name in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Branch(parent=name, child="a", time=1)

    def test_valid_branch(self):
        Branch(parent="a", child="b", time=10)
        # TODO: a branch at time=0 should probably be an error
        Branch(parent="a", child="b", time=0)

    def test_isclose(self):
        eps = 1e-50
        b1 = Branch(parent="a", child="b", time=1)
        self.assertTrue(b1.isclose(b1))
        self.assertTrue(b1.isclose(Branch(parent="a", child="b", time=1)))
        self.assertTrue(b1.isclose(Branch(parent="a", child="b", time=1 + eps)))

        self.assertFalse(b1.isclose(Branch(parent="x", child="b", time=1)))
        self.assertFalse(b1.isclose(Branch(parent="a", child="x", time=1)))
        self.assertFalse(b1.isclose(Branch(parent="b", child="a", time=1)))
        self.assertFalse(b1.isclose(Branch(parent="a", child="b", time=1 + 1e-9)))


class TestMerge(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

        for time in [-1e-12, -1, math.inf]:
            with self.assertRaises(ValueError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

    def test_bad_child(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child=name, time=1)

        for name in ("a", "b", "", "pop 1"):
            with self.assertRaises(ValueError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child=name, time=1)

    def test_bad_parents(self):
        for parents in (None, "b", {"b": 1}, set("b"), ("b", "b")):
            with self.assertRaises(TypeError):
                Merge(parents=parents, proportions=[0.5, 0.5], child="c", time=1)
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Merge(parents=["a", name], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(TypeError):
                Merge(parents=[name, "a"], proportions=[0.5, 0.5], child="c", time=1)

        for name in ("a", "c", "", "pop 1"):
            with self.assertRaises(ValueError):
                Merge(parents=["a", name], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(ValueError):
                Merge(parents=[name, "a"], proportions=[0.5, 0.5], child="c", time=1)
        with self.assertRaises(ValueError):
            Merge(parents=["a"], proportions=[1], child="b", time=1)

    def test_bad_proportions(self):
        for proportion in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Merge(parents=["a", "b"], child="c", time=1, proportions=[proportion])

        for proportion in (-10000, -1, -1e-9, 1.2, 100, math.inf):
            with self.assertRaises(ValueError):
                Merge(
                    parents=["a", "b"], child="c", time=1, proportions=[proportion, 0.5]
                )
            with self.assertRaises(ValueError):
                Merge(
                    parents=["a", "b"], child="c", time=1, proportions=[0.5, proportion]
                )

        with self.assertRaises(ValueError):
            Merge(parents=["a", "b"], proportions=[1], child="b", time=1)
        with self.assertRaises(ValueError):
            Merge(parents=["a", "b", "c"], proportions=[0.5, 0.5], child="b", time=1)
        with self.assertRaises(ValueError):
            Merge(
                parents=["a", "b"], proportions=[1 / 3, 1 / 3, 1 / 3], child="b", time=1
            )
        with self.assertRaises(ValueError):
            Merge(parents=["a", "b"], proportions=[0.1, 1], child="c", time=1)
        with self.assertRaises(ValueError):
            Merge(parents=["a", "b"], proportions=[-0.1, 1.1], child="c", time=1)
        with self.assertRaises(ValueError):
            Merge(parents=["a", "b"], proportions=[0.5], child="c", time=1)
        with self.assertRaises(ValueError):
            Merge(parents=["a", "b"], proportions=[1.0], child="c", time=1)
        with self.assertRaises(ValueError):
            Merge(
                parents=["a", "b", "c"], proportions=[0.5, 0.5, 0.5], child="d", time=1
            )

    def test_valid_merge(self):
        Merge(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=10)
        Merge(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=0)
        Merge(
            parents=["a", "b", "c"], proportions=[0.5, 0.25, 0.25], child="d", time=10
        )
        Merge(
            parents=["a", "b", "c"],
            proportions=[0.5, 0.5 - 1e-9, 1e-9],
            child="d",
            time=10,
        )
        Merge(parents=["a", "b"], proportions=[1 - 1e-9, 1e-9], child="c", time=10)

    def test_isclose(self):
        eps = 1e-50
        m1 = Merge(parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1)
        self.assertTrue(m1.isclose(m1))
        self.assertTrue(
            m1.isclose(
                Merge(parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1)
            )
        )
        self.assertTrue(
            m1.isclose(
                Merge(
                    parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1 + eps
                )
            )
        )
        self.assertTrue(
            m1.isclose(
                Merge(
                    parents=["a", "b"], proportions=[0.1 + eps, 0.9], child="c", time=1
                )
            )
        )
        self.assertTrue(
            m1.isclose(
                Merge(
                    parents=["a", "b"], proportions=[0.1, 0.9 + eps], child="c", time=1
                )
            )
        )
        # Order of parents/proportions doesn't matter.
        self.assertTrue(
            m1.isclose(
                Merge(parents=["b", "a"], proportions=[0.9, 0.1], child="c", time=1)
            )
        )

        self.assertFalse(
            m1.isclose(
                Merge(parents=["a", "x"], proportions=[0.1, 0.9], child="c", time=1)
            )
        )
        self.assertFalse(
            m1.isclose(
                Merge(parents=["x", "b"], proportions=[0.1, 0.9], child="c", time=1)
            )
        )
        self.assertFalse(
            m1.isclose(
                Merge(
                    parents=["a", "b"],
                    proportions=[0.1 + 1e-9, 0.9 - 1e-9],
                    child="c",
                    time=1,
                )
            )
        )
        self.assertFalse(
            m1.isclose(
                Merge(parents=["a", "b"], proportions=[0.1, 0.9], child="x", time=1)
            )
        )
        self.assertFalse(
            m1.isclose(
                Merge(
                    parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1 + 1e-9
                )
            )
        )
        self.assertFalse(
            m1.isclose(
                Merge(
                    parents=["a", "b", "x"],
                    proportions=[0.1, 0.9 - 1e-9, 1e-9],
                    child="c",
                    time=1,
                )
            )
        )


class TestAdmix(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

        for time in [-1e-12, -1, math.inf]:
            with self.assertRaises(ValueError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

    def test_bad_child(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child=name, time=1)

        for name in ("a", "b", "", "pop 1"):
            with self.assertRaises(ValueError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child=name, time=1)

    def test_bad_parents(self):
        for parents in (None, "b", {"b": 1}, set("b"), ("b", "b")):
            with self.assertRaises(TypeError):
                Admix(parents=parents, proportions=[0.5, 0.5], child="c", time=1)
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Admix(parents=["a", name], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(TypeError):
                Admix(parents=[name, "a"], proportions=[0.5, 0.5], child="c", time=1)

        for name in ("a", "c", "", "pop 1"):
            with self.assertRaises(ValueError):
                Admix(parents=["a", name], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(ValueError):
                Admix(parents=[name, "a"], proportions=[0.5, 0.5], child="c", time=1)
        with self.assertRaises(ValueError):
            Admix(parents=["a"], proportions=[1], child="b", time=1)

    def test_bad_proportions(self):
        for proportion in ("inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Admix(parents=["a", "b"], child="c", time=1, proportions=[proportion])

        for proportion in (-10000, -1, -1e-9, 1.2, 100, math.inf):
            with self.assertRaises(ValueError):
                Admix(
                    parents=["a", "b"], child="c", time=1, proportions=[proportion, 0.5]
                )
            with self.assertRaises(ValueError):
                Admix(
                    parents=["a", "b"], child="c", time=1, proportions=[0.5, proportion]
                )

        with self.assertRaises(ValueError):
            Admix(parents=["a", "b"], proportions=[1], child="b", time=1)
        with self.assertRaises(ValueError):
            Admix(parents=["a", "b", "c"], proportions=[0.5, 0.5], child="b", time=1)
        with self.assertRaises(ValueError):
            Admix(
                parents=["a", "b"], proportions=[1 / 3, 1 / 3, 1 / 3], child="b", time=1
            )
        with self.assertRaises(ValueError):
            Admix(parents=["a", "b"], proportions=[0.1, 1], child="c", time=1)
        with self.assertRaises(ValueError):
            Admix(parents=["a", "b"], proportions=[-0.1, 1.1], child="c", time=1)
        with self.assertRaises(ValueError):
            Admix(parents=["a", "b"], proportions=[0.5], child="c", time=1)
        with self.assertRaises(ValueError):
            Admix(parents=["a", "b"], proportions=[1.0], child="c", time=1)
        with self.assertRaises(ValueError):
            Admix(
                parents=["a", "b", "c"], proportions=[0.5, 0.5, 0.5], child="d", time=1
            )

    def test_valid_admixture(self):
        Admix(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=10)
        Admix(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=0)
        Admix(
            parents=["a", "b", "c"], proportions=[0.5, 0.25, 0.25], child="d", time=10
        )
        Admix(
            parents=["a", "b", "c"],
            proportions=[0.5, 0.5 - 1e-9, 1e-9],
            child="d",
            time=10,
        )
        Admix(parents=["a", "b"], proportions=[1 - 1e-9, 1e-9], child="c", time=10)

    def test_isclose(self):
        eps = 1e-50
        a1 = Admix(parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1)
        self.assertTrue(a1.isclose(a1))
        self.assertTrue(
            a1.isclose(
                Admix(parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1)
            )
        )
        self.assertTrue(
            a1.isclose(
                Admix(
                    parents=["a", "b"],
                    proportions=[0.1 + eps, 0.9],
                    child="c",
                    time=1 + eps,
                )
            )
        )
        self.assertTrue(
            a1.isclose(
                Admix(
                    parents=["a", "b"], proportions=[0.1 + eps, 0.9], child="c", time=1
                )
            )
        )
        self.assertTrue(
            a1.isclose(
                Admix(
                    parents=["a", "b"],
                    proportions=[0.1, 0.9 + eps],
                    child="c",
                    time=1 + eps,
                )
            )
        )
        # Order of parents/proportions doesn't matter.
        self.assertTrue(
            a1.isclose(
                Admix(parents=["b", "a"], proportions=[0.9, 0.1], child="c", time=1)
            )
        )

        self.assertFalse(
            a1.isclose(
                Admix(parents=["a", "x"], proportions=[0.1, 0.9], child="c", time=1)
            )
        )
        self.assertFalse(
            a1.isclose(
                Admix(parents=["x", "b"], proportions=[0.1, 0.9], child="c", time=1)
            )
        )
        self.assertFalse(
            a1.isclose(
                Admix(
                    parents=["a", "b"],
                    proportions=[0.1 + 1e-9, 0.9 - 1e-9],
                    child="c",
                    time=1,
                )
            )
        )
        self.assertFalse(
            a1.isclose(
                Admix(parents=["a", "b"], proportions=[0.1, 0.9], child="x", time=1)
            )
        )
        self.assertFalse(
            a1.isclose(
                Admix(
                    parents=["a", "b"], proportions=[0.1, 0.9], child="c", time=1 + 1e-9
                )
            )
        )
        self.assertFalse(
            a1.isclose(
                Admix(
                    parents=["a", "b", "x"],
                    proportions=[0.1, 0.9 - 1e-9, 1e-9],
                    child="c",
                    time=1,
                )
            )
        )


class TestDeme(unittest.TestCase):
    def test_properties(self):
        deme = Deme(
            name="a",
            description="b",
            ancestors=["c"],
            proportions=[1],
            start_time=math.inf,
            epochs=[
                Epoch(
                    start_time=math.inf,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )
            ],
        )
        self.assertEqual(deme.start_time, math.inf)
        self.assertEqual(deme.end_time, 0)
        self.assertEqual(deme.ancestors[0], "c")
        self.assertEqual(deme.proportions[0], 1)

        deme = Deme(
            name="a",
            description="b",
            ancestors=["c"],
            proportions=[1],
            start_time=100,
            epochs=[
                Epoch(
                    start_time=100,
                    end_time=50,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                ),
                Epoch(
                    start_time=50,
                    end_time=20,
                    start_size=100,
                    end_size=100,
                    size_function="constant",
                ),
                Epoch(
                    start_time=20,
                    end_time=1,
                    start_size=200,
                    end_size=1,
                    size_function="exponential",
                ),
            ],
        )
        self.assertEqual(deme.start_time, 100)
        self.assertEqual(deme.end_time, 1)

        deme = Deme(
            name="a",
            description=None,
            ancestors=["c"],
            proportions=[1],
            start_time=math.inf,
            epochs=[
                Epoch(
                    start_time=math.inf,
                    end_time=0,
                    start_size=1,
                    end_size=1,
                    size_function="constant",
                )
            ],
        )
        self.assertEqual(deme.description, None)

    def test_bad_id(self):
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Deme(
                    name=name,
                    description="b",
                    ancestors=[],
                    proportions=[],
                    start_time=math.inf,
                    epochs=[
                        Epoch(
                            start_time=math.inf,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )
        for name in ["", "501", "pop-1", "pop.2", "pop 3"]:
            with self.assertRaises(ValueError):
                Deme(
                    name=name,
                    description="b",
                    ancestors=[],
                    proportions=[],
                    start_time=math.inf,
                    epochs=[
                        Epoch(
                            start_time=math.inf,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )

    def test_bad_description(self):
        for description in (0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Deme(
                    name="a",
                    description=description,
                    ancestors=[],
                    proportions=[],
                    start_time=math.inf,
                    epochs=[
                        Epoch(
                            start_time=math.inf,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )
        with self.assertRaises(ValueError):
            Deme(
                name="a",
                description="",
                ancestors=[],
                proportions=[],
                start_time=math.inf,
                epochs=[
                    Epoch(
                        start_time=math.inf,
                        end_time=0,
                        start_size=1,
                        end_size=1,
                        size_function="constant",
                    )
                ],
            )

    def test_bad_ancestors(self):
        for ancestors in (None, "c", {}):
            with self.assertRaises(TypeError):
                Deme(
                    name="a",
                    description="b",
                    ancestors=ancestors,
                    proportions=[1],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )
        for name in (None, 0, math.inf, 1e3, {}, []):
            with self.assertRaises(TypeError):
                Deme(
                    name="a",
                    description="b",
                    ancestors=[name],
                    proportions=[1],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )
        for name in ["", "501", "pop-1", "pop.2", "pop 3"]:
            with self.assertRaises(ValueError):
                Deme(
                    name="a",
                    description="b",
                    ancestors=[name],
                    proportions=[1],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )

        with self.assertRaises(ValueError):
            Deme(
                name="a",
                description="b",
                ancestors=["a", "c"],
                proportions=[0.5, 0.5],
                start_time=10,
                epochs=[
                    Epoch(
                        start_time=10,
                        end_time=0,
                        start_size=1,
                        end_size=1,
                        size_function="constant",
                    )
                ],
            )
        with self.assertRaises(ValueError):
            # duplicate ancestors
            Deme(
                name="a",
                description="test",
                ancestors=["x", "x"],
                proportions=[0.5, 0.5],
                start_time=10,
                epochs=[
                    Epoch(
                        start_time=10,
                        end_time=0,
                        start_size=1,
                        end_size=1,
                        size_function="constant",
                    )
                ],
            )

    def test_bad_proportions(self):
        for proportions in (None, {}, 1e5, "proportions", math.nan):
            with self.assertRaises(TypeError):
                Deme(
                    name="a",
                    description="test",
                    ancestors=[],
                    proportions=proportions,
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )
        for proportion in (None, "inf", "100", {}, [], math.nan):
            with self.assertRaises(TypeError):
                Deme(
                    name="a",
                    description="test",
                    ancestors=["b"],
                    proportions=[proportion],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )

        for proportions in (
            [0.6, 0.7],
            [-0.5, 1.5],
            [0, 1.0],
            [0.5, 0.2, 0.3],
        ):
            with self.assertRaises(ValueError):
                Deme(
                    name="a",
                    description="test",
                    ancestors=["x", "y"],
                    proportions=proportions,
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )

        for proportion in (-10000, -1, -1e-9, 1.2, 100, math.inf):
            with self.assertRaises(ValueError):
                Deme(
                    name="a",
                    description="test",
                    ancestors=["b", "c"],
                    proportions=[0.5, proportion],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )
            with self.assertRaises(ValueError):
                Deme(
                    name="a",
                    description="test",
                    ancestors=["b", "c"],
                    proportions=[proportion, 0.5],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=0,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        )
                    ],
                )

    def test_epochs_out_of_order(self):
        for time in (5, -1, math.inf):
            with self.assertRaises(ValueError):
                Deme(
                    name="a",
                    description="b",
                    ancestors=["c"],
                    proportions=[1],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        ),
                        Epoch(
                            start_time=5,
                            end_time=time,
                            start_size=100,
                            end_size=100,
                            size_function="constant",
                        ),
                    ],
                )

    def test_epochs_are_a_partition(self):
        for start_time, end_time in [(math.inf, 100), (200, 100)]:
            with self.assertRaises(ValueError):
                Deme(
                    name="a",
                    description="b",
                    ancestors=["c"],
                    proportions=[1],
                    start_time=start_time,
                    epochs=[
                        Epoch(
                            start_time=start_time,
                            end_time=end_time,
                            start_size=1,
                            end_size=1,
                            size_function="constant",
                        ),
                        Epoch(
                            start_time=50,
                            end_time=0,
                            start_size=2,
                            end_size=2,
                            size_function="constant",
                        ),
                    ],
                )

    def test_bad_epochs(self):
        for epochs in (None, {}, "Epoch"):
            with self.assertRaises(TypeError):
                Deme(
                    name="a",
                    description="b",
                    ancestors=["c"],
                    proportions=[1],
                    start_time=10,
                    epochs=epochs,
                )

    def test_time_span(self):
        for start_time, end_time in zip((math.inf, 100, 20), (0, 20, 0)):
            deme = Deme(
                name="a",
                description="b",
                ancestors=["c"],
                proportions=[1],
                start_time=start_time,
                epochs=[
                    Epoch(
                        start_time=start_time,
                        end_time=end_time,
                        start_size=1,
                        end_size=1,
                        size_function="constant",
                    )
                ],
            )
            self.assertEqual(deme.time_span, start_time - end_time)
        with self.assertRaises(ValueError):
            deme = Deme(
                name="a",
                description="b",
                ancestors=["c"],
                proportions=[1],
                start_time=100,
                epochs=[
                    Epoch(
                        start_time=100,
                        end_time=100,
                        start_size=1,
                        end_size=1,
                        size_function="constant",
                    )
                ],
            )

    def test_isclose(self):
        d1 = Deme(
            name="a",
            description="foo deme",
            ancestors=[],
            proportions=[],
            start_time=10,
            epochs=[
                Epoch(
                    start_time=10,
                    end_time=5,
                    start_size=1,
                    end_size=1,
                    size_function="exponential",
                )
            ],
        )
        self.assertTrue(d1.isclose(d1))
        self.assertTrue(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )
        # Description field doesn't matter.
        self.assertTrue(
            d1.isclose(
                Deme(
                    name="a",
                    description="bar deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )

        #
        # Check inequalities.
        #

        self.assertFalse(
            d1.isclose(
                Deme(
                    name="b",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=["x"],
                    proportions=[1],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=9,
                    epochs=[
                        Epoch(
                            start_time=9,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=9,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )

        self.assertFalse(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=9,
                            end_size=1,
                            size_function="exponential",
                        )
                    ],
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                            selfing_rate=0.1,
                        )
                    ],
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    name="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    start_time=10,
                    epochs=[
                        Epoch(
                            start_time=10,
                            end_time=5,
                            start_size=1,
                            end_size=1,
                            size_function="exponential",
                            cloning_rate=0.1,
                        )
                    ],
                )
            )
        )


class TestGraph(unittest.TestCase):
    def test_bad_generation_time(self):
        for generation_time in ([], {}, "42", "inf", math.nan):
            with self.assertRaises(TypeError):
                Graph(
                    description="test",
                    time_units="years",
                    generation_time=generation_time,
                )
        for generation_time in (-100, -1e-9, 0, math.inf, None):
            with self.assertRaises(ValueError):
                Graph(
                    description="test",
                    time_units="years",
                    generation_time=generation_time,
                )

    def test_description(self):
        Graph(description="test", time_units="generations")
        Graph(description=None, time_units="generations")
        Graph(time_units="generations")

    def test_bad_description(self):
        for description in ([], {}, 0, 1e5, math.inf):
            with self.assertRaises(TypeError):
                Graph(
                    description=description,
                    time_units="generations",
                )
        with self.assertRaises(ValueError):
            Graph(
                description="",
                time_units="generations",
            )

    def test_doi(self):
        # We currently accept arbitrary strings in DOIs.
        # In any event here are some examples that should always be accepted.
        # https://www.doi.org/doi_handbook/2_Numbering.html
        for doi in [
            "10.1000/123456",
            "10.1000.10/123456",
            "10.1038/issn.1476-4687",
            # old doi proxy url; still supported
            "http://dx.doi.org/10.1006/jmbi.1998.2354",
            "https://dx.doi.org/10.1006/jmbi.1998.2354",
            # recommended doi proxy
            "http://doi.org/10.1006/jmbi.1998.2354",
            # https preferred
            "https://doi.org/10.1006/jmbi.1998.2354",
            # some symbols (e.g. #) must be encoded for the url to work
            "https://doi.org/10.1000/456%23789",
        ]:
            Graph(
                description="test",
                time_units="generations",
                doi=[doi],
            )

        # multiple DOIs
        Graph(
            description="test",
            time_units="generations",
            doi=[
                "10.1038/issn.1476-4687",
                "https://doi.org/10.1006/jmbi.1998.2354",
            ],
        )

        # empty list should also be fine
        Graph(
            description="test",
            time_units="generations",
            doi=[],
        )

    def test_bad_doi(self):
        for doi_list in ({}, "10.1000/123456", math.inf, 1e5, 0):
            with self.assertRaises(TypeError):
                Graph(
                    description="test",
                    time_units="generations",
                    doi=doi_list,
                )
        for doi in (None, {}, [], math.inf, 1e5, 0):
            with self.assertRaises(TypeError):
                Graph(
                    description="test",
                    time_units="generations",
                    doi=[doi],
                )

        with self.assertRaises(ValueError):
            Graph(
                description="test",
                time_units="generations",
                doi=[""],
            )

    def check_in_generations(self, dg1):
        assert dg1.generation_time is not None
        assert dg1.generation_time > 1
        dg1_copy = copy.deepcopy(dg1)
        dg2 = dg1.in_generations()
        # in_generations() shouldn't modify the original
        dg1_copy.assert_close(dg1)
        self.assertEqual(dg1.asdict(), dg1_copy.asdict())
        # but clearly dg2 should now differ
        assert not dg1.isclose(dg2)
        self.assertNotEqual(dg1.asdict(), dg2.asdict())

        # Alternate implementation, which recurses the object hierarchy.
        def in_generations2(dg):
            generation_time = dg.generation_time
            dg = copy.deepcopy(dg)
            dg.time_units = "generations"
            if generation_time is None:
                return dg
            dg.generation_time = None

            def divide_time_attrs(obj):
                if not hasattr(obj, "__dict__"):
                    return
                for name, value in obj.__dict__.items():
                    if name in ("time", "start_time", "end_time"):
                        if value is not None:
                            setattr(obj, name, value / generation_time)
                    elif isinstance(value, (list, tuple)):
                        for a in value:
                            divide_time_attrs(a)
                    else:
                        divide_time_attrs(value)

            divide_time_attrs(dg)
            return dg

        dg2.assert_close(in_generations2(dg1))
        self.assertEqual(in_generations2(dg1).asdict(), dg2.asdict())

        # in_generations2() shouldn't modify the original
        dg1.assert_close(dg1_copy)
        self.assertEqual(dg1.asdict(), dg1_copy.asdict())

        # in_generations() should be idempotent
        dg3 = dg2.in_generations()
        dg2.assert_close(dg3)
        self.assertEqual(dg2.asdict(), dg3.asdict())
        dg3 = in_generations2(dg2)
        dg2.assert_close(dg3)
        self.assertEqual(dg2.asdict(), dg3.asdict())

    def test_in_generations(self):
        examples_path = pathlib.Path(__file__).parent.parent / "examples"
        i = 0
        for yml in examples_path.glob("*.yml"):
            dg = load(yml)
            if dg.generation_time in (None, 1):
                # fake it
                dg.generation_time = 6
                dg.time_units = "years"
            self.check_in_generations(dg)
            i += 1
        self.assertGreater(i, 0)

    def test_isclose(self):
        b1 = Builder(description="test", time_units="generations")
        b2 = copy.deepcopy(b1)
        b1.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        g1 = b1.resolve()
        self.assertTrue(g1.isclose(g1))
        self.assertTrue(g1.isclose(demes.loads(demes.dumps(g1))))

        # Don't care about description for equality.
        b3 = Builder(description="some other description", time_units="generations")
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        g3 = b3.resolve()
        self.assertTrue(g1.isclose(g3))

        # Don't care about doi for equality.
        b3 = Builder(
            description="test",
            time_units="generations",
            doi=["https://example.com/foo.bar"],
        )
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        g3 = b3.resolve()
        self.assertTrue(g1.isclose(g3))

        # The order in which demes are added shouldn't matter.
        b3 = copy.deepcopy(b2)
        b4 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertTrue(g3.isclose(g4))

        # The order in which migrations are added shouldn't matter.
        b3.add_migration(source="d1", dest="d2", rate=1e-4, start_time=50, end_time=40)
        b3.add_migration(source="d2", dest="d1", rate=1e-5)
        b4.add_migration(source="d2", dest="d1", rate=1e-5)
        b4.add_migration(source="d1", dest="d2", rate=1e-4, start_time=50, end_time=40)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertTrue(g3.isclose(g4))

        # The order in which pulses are added shouldn't matter.
        b3.add_pulse(source="d1", dest="d2", proportion=0.01, time=100)
        b3.add_pulse(source="d1", dest="d2", proportion=0.01, time=50)
        b4.add_pulse(source="d1", dest="d2", proportion=0.01, time=50)
        b4.add_pulse(source="d1", dest="d2", proportion=0.01, time=100)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertTrue(g3.isclose(g4))

        # Order of symmetric migrations shouldn't matter, and neither should
        # the order of the demes lists within the migration objects.
        b3 = demes.Builder(defaults=dict(epoch=dict(start_size=1)))
        b3.add_deme("a")
        b3.add_deme("b")
        b3.add_deme("aa")
        b3.add_migration(demes=["a", "b"], rate=0.1)
        b3.add_migration(demes=["b", "aa"], rate=0.1)
        g3 = b3.resolve()
        b4 = demes.Builder(defaults=dict(epoch=dict(start_size=1)))
        b4.add_deme("a")
        b4.add_deme("b")
        b4.add_deme("aa")
        b4.add_migration(demes=["aa", "b"], rate=0.1)
        b4.add_migration(demes=["b", "a"], rate=0.1)
        g4 = b4.resolve()
        g3.assert_close(g4)

        #
        # Check inequalities
        #

        b3 = copy.deepcopy(b2)
        b3.add_deme("dX", epochs=[dict(start_size=1000, end_time=0)])
        g3 = b3.resolve()
        self.assertFalse(g1.isclose(g3))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1001, end_time=0)])
        g3 = b3.resolve()
        self.assertFalse(g1.isclose(g3))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        g3 = b3.resolve()
        self.assertFalse(g1.isclose(g3))

        b3 = copy.deepcopy(b1)
        b4 = copy.deepcopy(b1)
        b3.add_deme("dX", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme(
            "d2",
            ancestors=["dX"],
            start_time=50,
            epochs=[dict(start_size=1000, end_time=0)],
        )
        b4.add_deme("dX", epochs=[dict(start_size=1000)])
        b4.add_deme(
            "d2",
            ancestors=["d1"],
            start_time=50,
            epochs=[dict(start_size=1000, end_time=0)],
        )
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertFalse(g3.isclose(g4))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4 = copy.deepcopy(b2)
        b4.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_migration(source="d2", dest="d1", rate=1e-5)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertFalse(g3.isclose(g4))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_migration(source="d1", dest="d2", rate=1e-5)
        b4 = copy.deepcopy(b2)
        b4.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_migration(source="d2", dest="d1", rate=1e-5)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertFalse(g3.isclose(g4))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_migration(source="d2", dest="d1", rate=1e-5)
        b4 = copy.deepcopy(b2)
        b4.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_migration(demes=["d2", "d1"], rate=1e-5)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertFalse(g3.isclose(g4))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4 = copy.deepcopy(b2)
        b4.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_pulse(source="d1", dest="d2", proportion=0.01, time=100)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertFalse(g3.isclose(g4))

        b3 = copy.deepcopy(b2)
        b3.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b3.add_pulse(source="d2", dest="d1", proportion=0.01, time=100)
        b4 = copy.deepcopy(b2)
        b4.add_deme("d1", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_deme("d2", epochs=[dict(start_size=1000, end_time=0)])
        b4.add_pulse(source="d1", dest="d2", proportion=0.01, time=100)
        g3 = b3.resolve()
        g4 = b4.resolve()
        self.assertFalse(g3.isclose(g4))

        # symmetric migrations are not equivalent to asymmetric migrations
        b1 = Builder(defaults=dict(epoch=dict(start_size=1)))
        b1.add_deme("a")
        b1.add_deme("b")
        b1.add_migration(source="a", dest="b", rate=0.1)
        b2 = Builder(defaults=dict(epoch=dict(start_size=1)))
        b2.add_deme("a")
        b2.add_deme("b")
        b2.add_migration(demes=["a", "b"], rate=0.1)
        g1 = b1.resolve()
        g2 = b2.resolve()
        assert not g1.isclose(g2)

    def test_successors_predecessors(self):
        # single population
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        g = b.resolve()
        assert g.successors() == {"a": []}
        assert g.predecessors() == {"a": []}

        # successive branching
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b", ancestors=["a"], start_time=200)
        b.add_deme("c", ancestors=["b"], start_time=100)
        g = b.resolve()
        assert g.successors() == {"a": ["b"], "b": ["c"], "c": []}
        assert g.predecessors() == {"a": [], "b": ["a"], "c": ["b"]}

        # two successors
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b", ancestors=["a"], start_time=100)
        b.add_deme("c", ancestors=["a"], start_time=100)
        g = b.resolve()
        assert g.successors() == {"a": ["b", "c"], "b": [], "c": []}
        assert g.predecessors() == {"a": [], "b": ["a"], "c": ["a"]}

        # two predecessors
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme(
            "c", ancestors=["a", "b"], proportions=[1 / 2, 1 / 2], start_time=100
        )
        g = b.resolve()
        assert g.successors() == {"a": ["c"], "b": ["c"], "c": []}
        assert g.predecessors() == {"a": [], "b": [], "c": ["a", "b"]}

        # K3,3 graph (aka the utility graph)
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        ancestors = ["a", "b", "c"]
        children = ["x", "y", "z"]
        for name in ancestors:
            b.add_deme(name)
        for name in children:
            b.add_deme(
                name,
                ancestors=ancestors,
                proportions=[1 / 3, 1 / 3, 1 / 3],
                start_time=100,
            )
        g = b.resolve()
        assert g.successors() == {
            "a": children,
            "b": children,
            "c": children,
            "x": [],
            "y": [],
            "z": [],
        }
        assert g.predecessors() == {
            "a": [],
            "b": [],
            "c": [],
            "x": ancestors,
            "y": ancestors,
            "z": ancestors,
        }

        # pulses don't contribute
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_pulse(source="a", dest="b", proportion=0.1, time=100)
        g = b.resolve()
        assert g.successors() == {"a": [], "b": []}
        assert g.predecessors() == {"a": [], "b": []}

        # migrations don't contribute
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_migration(source="a", dest="b", rate=0.1)
        g = b.resolve()
        assert g.successors() == {"a": [], "b": []}
        assert g.predecessors() == {"a": [], "b": []}

    def dfsorted(self, objlist):
        """
        Return a copy of objlist that is sorted depth first. So lists nested
        within each object are first sorted, then the outer list of objects
        is sorted.
        """
        assert isinstance(objlist, list)
        objlist = copy.deepcopy(objlist)
        for obj in objlist:
            # This function will break if we ever change to using slotted classes,
            # because the __dict__ attribute won't be populated.
            assert not hasattr(obj, "__slots__")
            for name, attr in obj.__dict__.items():
                if isinstance(attr, list):
                    setattr(obj, name, sorted(attr))
        objlist.sort()
        return objlist

    def test_discrete_demographic_events(self):
        # unrelated populations
        b = Builder()
        for name in "abcde":
            b.add_deme(name, epochs=[dict(start_size=1)])
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("pulses", "splits", "branches", "mergers", "admixtures"):
            assert len(de[event]) == 0

        # pulse events
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.add_pulse(source="a", dest="b", time=100, proportion=0.1)
        b.add_pulse(source="a", dest="b", time=200, proportion=0.2)
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("splits", "branches", "mergers", "admixtures"):
            assert len(de[event]) == 0
        assert self.dfsorted(de["pulses"]) == self.dfsorted(
            [
                Pulse(source="a", dest="b", proportion=0.1, time=100),
                Pulse(source="a", dest="b", proportion=0.2, time=200),
            ]
        )

        # split event
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", ancestors=["a"])
        b.add_deme("c", ancestors=["a"])
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("pulses", "branches", "mergers", "admixtures"):
            assert len(de[event]) == 0
        assert self.dfsorted(de["splits"]) == self.dfsorted(
            [Split(parent="a", children=["b", "c"], time=100)]
        )

        # successive "splitting" (but with only one child)
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a", epochs=[dict(end_time=200)])
        b.add_deme("b", ancestors=["a"], start_time=200, epochs=[dict(end_time=100)])
        b.add_deme("c", ancestors=["b"], start_time=100)
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("pulses", "branches", "mergers", "admixtures"):
            assert len(de[event]) == 0
        assert self.dfsorted(de["splits"]) == self.dfsorted(
            [
                Split(parent="a", children=["b"], time=200),
                Split(parent="b", children=["c"], time=100),
            ]
        )

        # successive branches
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b", ancestors=["a"], start_time=200)
        b.add_deme("c", ancestors=["b"], start_time=100)
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("pulses", "splits", "mergers", "admixtures"):
            assert len(de[event]) == 0
        assert self.dfsorted(de["branches"]) == self.dfsorted(
            [
                Branch(parent="a", child="b", time=200),
                Branch(parent="b", child="c", time=100),
            ]
        )

        # merger event
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a", epochs=[dict(end_time=100)])
        b.add_deme("b", epochs=[dict(end_time=100)])
        b.add_deme(
            "c", ancestors=["a", "b"], proportions=[1 / 2, 1 / 2], start_time=100
        )
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("pulses", "splits", "branches", "admixtures"):
            assert len(de[event]) == 0
        assert self.dfsorted(de["mergers"]) == self.dfsorted(
            [Merge(parents=["a", "b"], child="c", proportions=[1 / 2, 1 / 2], time=100)]
        )

        # admixture event
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b", epochs=[dict(end_time=100)])
        b.add_deme(
            "c", ancestors=["a", "b"], proportions=[1 / 2, 1 / 2], start_time=100
        )
        g = b.resolve()
        de = g.discrete_demographic_events()
        assert len(de) == 5
        for event in ("pulses", "splits", "branches", "mergers"):
            assert len(de[event]) == 0
        assert self.dfsorted(de["admixtures"]) == self.dfsorted(
            [Admix(parents=["a", "b"], child="c", proportions=[1 / 2, 1 / 2], time=100)]
        )


class TestGraphResolution(unittest.TestCase):
    def test_basic_resolution(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.resolve()

        b = Builder()
        b.add_deme("a", epochs=[dict(end_size=1)])
        b.resolve()

        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1, end_time=100)])
        b.add_deme("b", ancestors=["a"], epochs=[dict(start_size=1)])
        b.resolve()

        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.add_migration(source="a", dest="b", rate=1e-5)
        b.resolve()

        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.add_deme("c", epochs=[dict(start_size=1)])
        b.add_migration(demes=["a", "b", "c"], rate=1e-5)
        b.resolve()

        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.add_pulse(source="a", dest="b", proportion=0.1, time=100)
        b.resolve()

    def test_bad_data_dict(self):
        # not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.data = data
            with pytest.raises(TypeError):
                b.resolve()

        # empty dict
        b = Builder.fromdict(dict())
        with pytest.raises(KeyError):
            b.resolve()

    def test_bad_toplevel(self):
        # no time units
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        del b.data["time_units"]
        with pytest.raises(KeyError):
            b.resolve()

        # unrecognised toplevel field
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.data["thyme_younerts"] = "generations"
        with pytest.raises(KeyError):
            b.resolve()

        # unrecognised toplevel fields
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.data["foo"] = [dict(bar=10), dict(baz=20)]
        b.data["zort"] = "tron"
        with pytest.raises(KeyError):
            b.resolve()

    def test_bad_toplevel_defaults(self):
        # defaults is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.data["defaults"] = data
            with pytest.raises(TypeError):
                b.resolve()

        # unrecognised defaults fields
        b = Builder(defaults=dict(rate=0.1, proportion=0.1))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError):
            b.resolve()

        # unrecognised defaults.deme fields
        b = Builder(defaults=dict(deme=dict(foo=10, bar=20)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError):
            b.resolve()

        # unrecognised defaults.epoch fields
        b = Builder(defaults=dict(epoch=dict(foo=10)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError):
            b.resolve()

        # unrecognised defaults.migration fields
        b = Builder(defaults=dict(migration=dict(foo=10)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError):
            b.resolve()

        # unrecognised defaults.pulse fields
        b = Builder(defaults=dict(pulse=dict(foo={}, bar=[], baz=None)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError):
            b.resolve()

        # defaults field is not a dict
        for data in (None, [], "string", 0, 1e-5):
            for key in ("deme", "migration", "pulse", "epoch"):
                b = Builder()
                b.add_deme("a", epochs=[dict(start_size=1)])
                b.data["defaults"] = {key: data}
                with pytest.raises(TypeError):
                    b.resolve()

    def test_bad_demelevel_defaults(self):
        # defaults is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.data["demes"][0]["defaults"] = data
            with pytest.raises(TypeError):
                b.resolve()

        # unrecognised defaults fields
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)], defaults=dict(foo=10, bar=20))
        with pytest.raises(KeyError):
            b.resolve()

        # defaults.epoch field is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)], defaults=data)
            b.data["demes"][0]["defaults"] = dict(epoch=data)
            with pytest.raises(TypeError):
                b.resolve()

        # unrecognised defaults.epoch fields
        b = Builder()
        b.add_deme(
            "a", epochs=[dict(start_size=1)], defaults=dict(epoch=dict(foo=10, bar=20))
        )
        with pytest.raises(KeyError):
            b.resolve()

    def test_bad_demes(self):
        # no demes
        b = Builder()
        with pytest.raises(KeyError):
            b.resolve()

        # demes is not a list
        for data in (None, {}, "string", 0, 1e-5):
            b = Builder()
            b.data["demes"] = data
            with pytest.raises(TypeError):
                b.resolve()

        # deme is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.data["demes"] = [data]
            with pytest.raises(TypeError):
                b.resolve()

        # deme has no name
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        del b.data["demes"][0]["name"]
        with pytest.raises(KeyError):
            b.resolve()

        # no epochs given
        b = Builder()
        b.add_deme("a")
        with self.assertRaises(KeyError):
            b.resolve()

        # missing start_size or end_size
        b = Builder()
        b.add_deme("a", epochs=[dict(end_time=1)])
        with self.assertRaises(KeyError):
            b.resolve()

        # ancestors must be a list
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme(
            "b",
            ancestors="a",
            start_time=10,
            epochs=[dict(start_size=1)],
        )
        with self.assertRaises(TypeError):
            b.resolve()

        # ancestor x doesn't exist
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme(
            "b",
            ancestors=["a", "x"],
            proportions=[0.5, 0.5],
            epochs=[dict(start_size=1)],
        )
        with self.assertRaises(ValueError):
            b.resolve()

    def test_duplicate_deme(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("a", epochs=[dict(start_size=1)])
        with self.assertRaises(ValueError):
            b.resolve()

    def test_duplicate_ancestors(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, end_time=50)])
        b.add_deme(
            "b",
            ancestors=["a", "a"],
            proportions=[0.5, 0.5],
            start_time=100,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b.resolve()

    def test_bad_start_time_wrt_ancestors(self):
        b1 = Builder()
        b1.add_deme("ancestral", epochs=[dict(start_size=100)])
        b1.add_deme(
            "a",
            start_time=100,
            ancestors=["ancestral"],
            epochs=[dict(start_size=100, end_time=50)],
        )
        b1.add_deme("b", epochs=[dict(start_size=100)])

        # start_time too old
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "c",
            ancestors=["a"],
            start_time=200,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # start_time too young
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "c",
            ancestors=["a"],
            start_time=20,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # start_time too old
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "c",
            ancestors=["a", "b"],
            proportions=[0.5, 0.5],
            start_time=200,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # start_time too young
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "c",
            ancestors=["a", "b"],
            proportions=[0.5, 0.5],
            start_time=20,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # start_time not provided
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "c",
            ancestors=["a", "b"],
            proportions=[0.5, 0.5],
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # finite start time, but no ancestors
        b2 = copy.deepcopy(b1)
        b2.add_deme("c", start_time=100, epochs=[dict(start_size=100)])
        with self.assertRaises(ValueError):
            b2.resolve()

    def test_proportions(self):
        b1 = Builder()
        b1.add_deme("a", epochs=[dict(start_size=100, end_time=50)])
        b1.add_deme("b", epochs=[dict(start_size=100, end_time=50)])

        # proportions missing
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "c",
            ancestors=["a", "b"],
            start_time=100,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        # proportions wrong length
        b2.add_deme(
            "c",
            ancestors=["a", "b"],
            proportions=[1],
            start_time=100,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        # proportions wrong length
        b2.add_deme(
            "c",
            ancestors=["a", "b"],
            proportions=[1 / 3, 1 / 3, 1 / 3],
            start_time=100,
            epochs=[dict(start_size=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        b2.add_deme("c", ancestors=["b"], epochs=[dict(start_size=100)])
        g2 = b2.resolve()
        self.assertEqual(len(g2["c"].proportions), 1)
        self.assertEqual(g2["c"].proportions[0], 1.0)

    def test_deme_end_time(self):
        b1 = Builder()
        b1.add_deme(
            "a",
            epochs=[
                dict(end_time=100, start_size=10),
                dict(start_size=20, end_time=0),
            ],
        )

        # can't have epoch end_time == deme start_time
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "b",
            start_time=100,
            ancestors=["a"],
            epochs=[dict(start_size=100, end_time=100)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # can't have epoch end_time > deme start_time
        b2 = copy.deepcopy(b1)
        b2.add_deme(
            "b",
            start_time=100,
            ancestors=["a"],
            epochs=[dict(start_size=100, end_time=200)],
        )
        with self.assertRaises(ValueError):
            b2.resolve()

        # Check that end_time can be ommitted for final epoch
        b2 = copy.deepcopy(b1)
        b2.add_deme("x", start_time=100, ancestors=["a"], epochs=[dict(start_size=100)])
        b2.add_deme("y", epochs=[dict(start_size=100)])
        b2.add_deme(
            "z",
            start_time=100,
            ancestors=["a"],
            epochs=[dict(start_size=100, end_time=10), dict(start_size=10)],
        )
        b2.resolve()

    def test_bad_epochs(self):
        # deme has no epochs
        b = Builder()
        b.add_deme("a")
        with pytest.raises(KeyError):
            b.resolve()

        # epochs is not a list
        for data in (None, {}, "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a")
            b.data["demes"][0]["epochs"] = data
            with pytest.raises(TypeError):
                b.resolve()

        # epoch is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a")
            b.data["demes"][0]["epochs"] = [data]
            with pytest.raises(TypeError):
                b.resolve()

        # epochs out of order
        b = Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=1, end_time=0),
                dict(start_size=1, end_time=100),
            ],
        )
        with self.assertRaises(ValueError):
            b.resolve()

        # must have a start_size or end_size
        b = Builder()
        b.add_deme("a", epochs=[dict(end_time=0)])
        with self.assertRaises(KeyError):
            b.resolve()

        # except for the last epoch, all epochs must have an end_time
        b = Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=1, end_time=100),
                dict(start_size=1),
                dict(start_size=1, end_time=0),
            ],
        )
        with self.assertRaises(KeyError):
            b.resolve()

        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=10), dict(end_time=0, start_size=100)])
        with self.assertRaises(KeyError):
            b.resolve()

    def test_bad_migrations(self):
        # migrations is not a list
        for data in (None, {}, "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.data["migrations"] = data
            with pytest.raises(TypeError):
                b.resolve()

        # migration is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.data["migrations"] = [data]
            with pytest.raises(TypeError):
                b.resolve()

        # missing required migration field: 'rate' (asymmetric)
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.data["migrations"] = [dict(source="a", dest="b")]
        with pytest.raises(KeyError):
            b.resolve()

        # missing required migration field: 'rate' (symmetric)
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.data["migrations"] = [dict(demes=["a", "b"])]
        with pytest.raises(KeyError):
            b.resolve()

        # unable to determine if symmetric or asymmetric
        for migration in [
            dict(source="a", rate=0.1),
            dict(dest="a", rate=0.1),
            dict(demes=["a", "b"], dest="a", rate=0.1),
            dict(demes=["a", "b"], source="a", rate=0.1),
        ]:
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.add_deme("b", epochs=[dict(start_size=1)])
            b.data["migrations"] = [migration]
            with pytest.raises(KeyError):
                b.resolve()

        # no demes participating in migration
        b = Builder()
        b.add_deme("X", epochs=[dict(start_size=100)])
        b.add_migration(demes=[], rate=0)
        with self.assertRaises(ValueError):
            b.resolve()

        # only one deme participating in migration
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100)])
        b.add_migration(demes=["a"], rate=0.1)
        with self.assertRaises(ValueError):
            b.resolve()

        # source and dest aren't in the graph
        b = Builder()
        b.add_deme("X", epochs=[dict(start_size=100)])
        b.add_migration(source="a", dest="b", rate=0.1)
        with self.assertRaises(ValueError):
            b.resolve()

        # dest not in graph
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100)])
        b.add_migration(source="a", dest="b", rate=0.1)
        with self.assertRaises(ValueError):
            b.resolve()

        # source not in graph
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100)])
        b.add_migration(source="b", dest="a", rate=0.1)
        with self.assertRaises(ValueError):
            b.resolve()

    def test_bad_migration_time(self):
        b = Builder()
        b.add_deme("deme1", epochs=[dict(start_size=1000, end_time=0)])
        b.add_deme("deme2", epochs=[dict(end_time=100, start_size=1000)])
        b.add_migration(
            source="deme1", dest="deme2", rate=0.01, start_time=1000, end_time=0
        )
        with self.assertRaises(ValueError):
            b.resolve()

    def test_overlapping_migrations(self):
        b1 = Builder()
        b1.add_deme("A", epochs=[dict(start_size=1)])
        b1.add_deme("B", epochs=[dict(start_size=1)])
        b1.add_migration(source="A", dest="B", rate=0.01)

        b2 = copy.deepcopy(b1)
        b2.add_migration(source="A", dest="B", start_time=10, rate=0.02)
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        b2.add_migration(demes=["A", "B"], rate=0.02)
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        b2.add_migration(demes=["A", "B"], rate=0.02, end_time=100)
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        b2.add_migration(source="B", dest="A", rate=0.03, start_time=100, end_time=10)
        b2.add_migration(source="B", dest="A", rate=0.04, start_time=50)
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        b2.add_migration(source="B", dest="A", rate=0.03, start_time=5)
        b2.add_migration(source="B", dest="A", rate=0.05, start_time=10)
        with self.assertRaises(ValueError):
            b2.resolve()

        b2 = copy.deepcopy(b1)
        b2.add_migration(source="B", dest="A", rate=0.01, start_time=10, end_time=5)
        b2.resolve()

    def test_bad_migration_rates_sum(self):
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("A")
        b.add_deme("B")
        b.add_deme("C")
        b.add_migration(source="B", dest="A", rate=0.5)
        b.add_migration(source="C", dest="A", rate=0.5)
        # rates into A sum to 1.0, which is fine
        b.resolve()

        b.add_deme("D")
        b.add_migration(source="D", dest="A", rate=1e-9)
        with pytest.raises(ValueError, match="sum of migration rates"):
            b.resolve()

        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("A")
        b.add_deme("B")
        b.add_deme("C")
        b.add_migration(source="C", dest="A", rate=0.6, start_time=100, end_time=50)
        b.add_migration(source="B", dest="A", rate=0.6, start_time=200, end_time=100)
        # migration time intervals don't intersect, so this is fine
        b.resolve()

        b.add_deme("D")
        b.add_migration(source="D", dest="A", rate=0.6, start_time=60, end_time=20)
        with pytest.raises(ValueError, match="sum of migration rates"):
            b.resolve()

        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("A")
        b.add_deme("B")
        b.add_deme("C")
        b.add_migration(demes=["A", "B"], rate=0.6, start_time=100)
        b.add_migration(source="C", dest="A", rate=0.6)
        with pytest.raises(ValueError, match="sum of migration rates"):
            b.resolve()

    def test_bad_pulses(self):
        # pulses is not a list
        for data in (None, {}, "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.data["pulses"] = data
            with pytest.raises(TypeError):
                b.resolve()

        # pulse is not a dict
        for data in (None, [], "string", 0, 1e-5):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.data["pulses"] = [data]
            with pytest.raises(TypeError):
                b.resolve()

        # dest not in graph
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, end_time=0)])
        b.add_pulse(source="a", dest="b", proportion=0.1, time=10)
        with self.assertRaises(ValueError):
            b.resolve()

        # source not in graph
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, end_time=0)])
        b.add_pulse(source="b", dest="a", proportion=0.1, time=10)
        with self.assertRaises(ValueError):
            b.resolve()

        for field in ("source", "dest", "time", "proportion"):
            b = Builder()
            b.add_deme("a", epochs=[dict(start_size=1)])
            b.add_deme("b", epochs=[dict(start_size=1)])
            b.add_pulse(source="a", dest="b", proportion=0.1, time=100)
            del b.data["pulses"][0][field]
            with pytest.raises(KeyError):
                b.resolve()

    def test_bad_pulse_time(self):
        b = Builder()
        b.add_deme("deme1", epochs=[dict(start_size=1000, end_time=0)])
        b.add_deme("deme2", epochs=[dict(end_time=100, start_size=1000)])
        b.add_pulse(source="deme1", dest="deme2", proportion=0.1, time=10)
        with self.assertRaises(ValueError):
            b.resolve()

        b = Builder(defaults=dict(epoch=dict(start_size=100)))
        b.add_deme("A")
        b.add_deme("B", start_time=100, ancestors=["A"], epochs=[dict(end_time=50)])
        g = b.resolve()

        # Can't have pulse at the dest deme's end_time.
        b2 = copy.deepcopy(b)
        b2.add_pulse(source="A", dest="B", time=g["B"].end_time, proportion=0.1)
        with self.assertRaises(ValueError):
            b2.resolve()

        # Can't have pulse at the source deme's start_time.
        b2 = copy.deepcopy(b)
        b2.add_pulse(source="B", dest="A", time=g["B"].start_time, proportion=0.1)
        with self.assertRaises(ValueError):
            b2.resolve()

    def test_pulse_same_time(self):
        b1 = Builder()
        for j in range(4):
            b1.add_deme(f"d{j}", epochs=[dict(start_size=1000)])

        T = 100  # time of pulses

        # Warn for duplicate pulses
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d1", time=T, proportion=0.1)
        b2.add_pulse(source="d0", dest="d1", time=T, proportion=0.1)
        with pytest.warns(UserWarning):
            b2.resolve()

        # Warn for: d0 -> d1; d1 -> d2.
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d1", time=T, proportion=0.1)
        b2.add_pulse(source="d1", dest="d2", time=T, proportion=0.1)
        with pytest.warns(UserWarning):
            b2.resolve()

        # Warn for: d0 -> d2; d1 -> d2.
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d2", time=T, proportion=0.1)
        b2.add_pulse(source="d1", dest="d2", time=T, proportion=0.1)
        with pytest.warns(UserWarning):
            b2.resolve()

        # Shouldn't warn for: d0 -> d1; d0 -> d2.
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d1", time=T, proportion=0.1)
        b2.add_pulse(source="d0", dest="d2", time=T, proportion=0.1)
        with pytest.warns(None) as record:
            b2.resolve()
        assert len(record) == 0

        # Shouldn't warn for: d0 -> d1; d2 -> d3.
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d1", time=T, proportion=0.1)
        b2.add_pulse(source="d2", dest="d3", time=T, proportion=0.1)
        with pytest.warns(None) as record:
            b2.resolve()
        assert len(record) == 0

        # Different pulse times shouldn't warn for: d0 -> d1; d1 -> d2.
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d1", time=T, proportion=0.1)
        b2.add_pulse(source="d1", dest="d2", time=2 * T, proportion=0.1)
        with pytest.warns(None) as record:
            b2.resolve()
        assert len(record) == 0

        # Different pulse times shouldn't warn for: d0 -> d2; d1 -> d2.
        b2 = copy.deepcopy(b1)
        b2.add_pulse(source="d0", dest="d2", time=T, proportion=0.1)
        b2.add_pulse(source="d1", dest="d2", time=2 * T, proportion=0.1)
        with pytest.warns(None) as record:
            b2.resolve()
        assert len(record) == 0

    def test_pulse_proportions_sum(self):
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        b.add_deme("a")
        b.add_deme("b")
        b.add_deme("c")
        b.add_pulse(source="b", dest="a", time=100, proportion=0.6)
        b.add_pulse(source="c", dest="a", time=100, proportion=0.6)
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError):
                b.resolve()

    def test_toplevel_defaults_deme(self):
        # description
        b = Builder(defaults=dict(deme=dict(description="Demey MacDemeFace")))
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)], description="deme bee")
        g = b.resolve()
        assert g["a"].description == "Demey MacDemeFace"
        assert g["b"].description == "deme bee"

        # start_time
        b = Builder(defaults=dict(deme=dict(start_time=100)))
        b.add_deme("a", epochs=[dict(start_size=1)], start_time=math.inf)
        b.add_deme("b", epochs=[dict(start_size=1)], ancestors=["a"])
        g = b.resolve()
        assert math.isinf(g["a"].start_time)
        assert g["b"].start_time == 100

        # ancestors
        b = Builder(defaults=dict(deme=dict(ancestors=["a"])))
        b.add_deme("a", epochs=[dict(start_size=1)], ancestors=[])
        b.add_deme("b", epochs=[dict(start_size=1)], start_time=100)
        g = b.resolve()
        assert g["a"].ancestors == []
        assert g["b"].ancestors == ["a"]

        # proportions
        b = Builder(defaults=dict(deme=dict(proportions=[0.1, 0.9])))
        b.add_deme("a", epochs=[dict(start_size=1)], proportions=[])
        b.add_deme("b", epochs=[dict(start_size=1)], proportions=[])
        b.add_deme(
            "c", epochs=[dict(start_size=1)], ancestors=["a", "b"], start_time=100
        )
        g = b.resolve()
        assert g["a"].proportions == g["b"].proportions == []
        assert g["c"].proportions == [0.1, 0.9]

        # proportions and ancestors
        b = Builder(
            defaults=dict(
                deme=dict(ancestors=["a", "b", "c"], proportions=[0.1, 0.7, 0.2])
            )
        )
        for name in "abc":
            b.add_deme(name, epochs=[dict(start_size=1)], ancestors=[], proportions=[])
        for name in "xyz":
            b.add_deme(name, epochs=[dict(start_size=1)], start_time=100)
        g = b.resolve()
        assert g["a"].ancestors == g["b"].ancestors == g["c"].ancestors == []
        assert g["a"].proportions == g["b"].proportions == g["c"].proportions == []
        assert (
            g["x"].ancestors == g["y"].ancestors == g["z"].ancestors == ["a", "b", "c"]
        )
        assert (
            g["x"].proportions
            == g["y"].proportions
            == g["z"].proportions
            == [0.1, 0.7, 0.2]
        )

    def test_toplevel_defaults_migration(self):
        # rate
        b = Builder(defaults=dict(migration=dict(rate=0.1)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.add_deme("b", epochs=[dict(start_size=1)])
        b.add_migration(source="a", dest="b", start_time=300, end_time=200)
        b.add_migration(source="a", dest="b", start_time=200, end_time=100, rate=0.2)
        b.add_migration(demes=["a", "b"], start_time=30, end_time=20)
        b.add_migration(demes=["a", "b"], start_time=20, end_time=10, rate=0.2)
        g = b.resolve()
        assert g.migrations[0].rate == 0.1
        assert g.migrations[1].rate == 0.2
        assert g.migrations[2].rate == 0.1
        assert g.migrations[3].rate == 0.2

        # start_time
        b = Builder(defaults=dict(migration=dict(start_time=100)))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_migration(source="a", dest="b", end_time=90, rate=0.1)
        b.add_migration(source="b", dest="a", start_time=90, end_time=80, rate=0.1)
        b.add_migration(demes=["c", "d"], end_time=90, rate=0.1)
        b.add_migration(demes=["c", "d"], start_time=90, end_time=80, rate=0.1)
        g = b.resolve()
        assert g.migrations[0].start_time == 100
        assert g.migrations[1].start_time == 90
        assert g.migrations[2].start_time == 100
        assert g.migrations[3].start_time == 90

        # end_time
        b = Builder(defaults=dict(migration=dict(end_time=100)))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_migration(source="a", dest="b", start_time=200, rate=0.1)
        b.add_migration(source="a", dest="b", start_time=100, end_time=50, rate=0.1)
        b.add_migration(demes=["c", "d"], start_time=200, rate=0.1)
        b.add_migration(demes=["c", "d"], start_time=100, end_time=50, rate=0.1)
        g = b.resolve()
        assert g.migrations[0].end_time == 100
        assert g.migrations[1].end_time == 50
        assert g.migrations[2].end_time == 100
        assert g.migrations[3].end_time == 50

        # source
        b = Builder(defaults=dict(migration=dict(source="a")))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        for name in "bcd":
            b.add_migration(dest=name, rate=0.1)
        b.add_migration(source="d", dest="a", rate=0.2)
        g = b.resolve()
        assert (
            g.migrations[0].source
            == g.migrations[1].source
            == g.migrations[2].source
            == "a"
        )
        assert g.migrations[3].source == "d"
        # source still defaults to "a", but we want symmetric migration
        for name in "xyz":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_migration(source=None, demes=["x", "y", "z"], rate=0.1)
        g = b.resolve()
        assert isinstance(g.migrations[4], SymmetricMigration)
        assert g.migrations[4].demes == ["x", "y", "z"]

        # dest
        b = Builder(defaults=dict(migration=dict(dest="a")))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        for name in "bcd":
            b.add_migration(source=name, rate=0.1)
        b.add_migration(source="a", dest="d", rate=0.2)
        g = b.resolve()
        assert (
            g.migrations[0].dest == g.migrations[1].dest == g.migrations[2].dest == "a"
        )
        assert g.migrations[3].dest == "d"
        # dest still defaults to "a", but we want symmetric migration
        for name in "xyz":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_migration(dest=None, demes=["x", "y", "z"], rate=0.1)
        g = b.resolve()
        assert isinstance(g.migrations[4], SymmetricMigration)
        assert g.migrations[4].demes == ["x", "y", "z"]

        # demes
        b = Builder(defaults=dict(migration=dict(demes=["a", "b", "c"])))
        for name in "abc":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_migration(start_time=200, end_time=100, rate=0.1)
        b.add_migration(demes=["a", "b"], start_time=100, end_time=0, rate=0.2)
        g = b.resolve()
        assert g.migrations[0].demes == ["a", "b", "c"]
        assert g.migrations[1].demes == ["a", "b"]
        # demes still defaults to ["a", "b", "c"], but we want asymmetric migration
        for name in "xy":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_migration(demes=None, source="x", dest="y", rate=0.3)
        g = b.resolve()
        assert isinstance(g.migrations[2], AsymmetricMigration)
        assert g.migrations[2].source == "x"
        assert g.migrations[2].dest == "y"

    def test_toplevel_defaults_pulse(self):
        # source
        b = Builder(defaults=dict(pulse=dict(source="a")))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        for name in "bcd":
            b.add_pulse(dest=name, proportion=0.1, time=100)
        b.add_pulse(source="d", dest="a", proportion=0.2, time=200)
        g = b.resolve()
        assert g.pulses[0].source == g.pulses[1].source == g.pulses[2].source == "a"
        assert g.pulses[3].source == "d"

        # dest
        b = Builder(defaults=dict(pulse=dict(dest="a")))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        for name in "bcd":
            b.add_pulse(source=name, proportion=0.1, time=100)
        b.add_pulse(dest="d", source="a", proportion=0.2, time=200)
        with pytest.warns(UserWarning):
            g = b.resolve()
        assert g.pulses[0].dest == g.pulses[1].dest == g.pulses[2].dest == "a"
        assert g.pulses[3].dest == "d"

        # time
        b = Builder(defaults=dict(pulse=dict(time=100)))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        for name in "bcd":
            b.add_pulse(source="a", dest=name, proportion=0.1)
        b.add_pulse(source="d", dest="a", proportion=0.2, time=50)
        g = b.resolve()
        assert g.pulses[0].time == g.pulses[1].time == g.pulses[2].time == 100
        assert g.pulses[3].time == 50

        # proportion
        b = Builder(defaults=dict(pulse=dict(proportion=0.1)))
        for name in "abcd":
            b.add_deme(name, epochs=[dict(start_size=1)])
        for name in "bcd":
            b.add_pulse(source="a", dest=name, time=100)
        b.add_pulse(source="d", dest="a", time=50, proportion=0.2)
        g = b.resolve()
        assert (
            g.pulses[0].proportion
            == g.pulses[1].proportion
            == g.pulses[2].proportion
            == 0.1
        )
        assert g.pulses[3].proportion == 0.2

    # Test toplevel epoch defaults, including overrides.
    def test_toplevel_defaults_epoch(self):
        # start_size
        b = Builder(defaults=dict(epoch=dict(start_size=1)))
        for name in "abc":
            b.add_deme(name)
        b.add_deme("d", epochs=[dict(start_size=2)])
        b.add_deme(
            "e",
            epochs=[
                dict(end_time=100),
                dict(end_time=50, end_size=99),
                dict(start_size=3),
            ],
        )
        b.add_deme(
            "f",
            defaults=dict(epoch=dict(start_size=4)),
            epochs=[
                dict(end_time=100),
                dict(end_time=50, end_size=99),
                dict(start_size=5, end_size=99),
            ],
        )
        g = b.resolve()
        assert (
            g["a"].epochs[0].start_size
            == g["b"].epochs[0].start_size
            == g["c"].epochs[0].start_size
            == 1
        )
        assert g["d"].epochs[0].start_size == 2
        assert g["e"].epochs[0].start_size == 1
        assert g["e"].epochs[1].start_size == 1
        assert g["e"].epochs[2].start_size == 3
        assert g["f"].epochs[0].start_size == 4
        assert g["f"].epochs[1].start_size == 4
        assert g["f"].epochs[2].start_size == 5

        # end_size
        b = Builder(defaults=dict(epoch=dict(end_size=1)))
        for name in "abc":
            b.add_deme(name)
        b.add_deme("d", epochs=[dict(end_size=2)])
        b.add_deme(
            "e",
            epochs=[
                dict(end_time=100),
                dict(end_time=50, start_size=99),
                dict(end_size=3),
            ],
        )
        b.add_deme(
            "f",
            defaults=dict(epoch=dict(end_size=4)),
            epochs=[
                dict(end_time=100),
                dict(end_time=50, start_size=99),
                dict(start_size=99, end_size=5),
            ],
        )
        g = b.resolve()
        assert (
            g["a"].epochs[0].end_size
            == g["b"].epochs[0].end_size
            == g["c"].epochs[0].end_size
            == 1
        )
        assert g["d"].epochs[0].end_size == 2
        assert g["e"].epochs[0].end_size == 1
        assert g["e"].epochs[1].end_size == 1
        assert g["e"].epochs[2].end_size == 3
        assert g["f"].epochs[0].end_size == 4
        assert g["f"].epochs[1].end_size == 4
        assert g["f"].epochs[2].end_size == 5

        # end_time
        b = Builder(defaults=dict(epoch=dict(end_time=100)))
        for name in "abc":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_deme(
            "d",
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[dict(start_size=1, end_time=50), dict(start_size=2, end_time=0)],
        )
        b.add_deme(
            "e",
            defaults=dict(epoch=dict(end_time=50)),  # this is silly
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[dict(start_size=1), dict(start_size=2, end_time=10)],
        )
        g = b.resolve()
        assert g["a"].end_time == g["b"].end_time == g["c"].end_time == 100
        assert g["d"].end_time == 0
        assert g["d"].epochs[0].end_time == 50
        assert g["d"].epochs[1].end_time == 0
        assert g["e"].end_time == 10
        assert g["e"].epochs[0].end_time == 50
        assert g["e"].epochs[1].end_time == 10

        # selfing_rate
        b = Builder(defaults=dict(epoch=dict(selfing_rate=0.1)))
        for name in "abc":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_deme(
            "d",
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[
                dict(start_size=1, end_time=50),
                dict(start_size=1, selfing_rate=0),
            ],
        )
        b.add_deme(
            "e",
            defaults=dict(epoch=dict(selfing_rate=0.2)),
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[
                dict(start_size=1, end_time=90),
                dict(start_size=1, end_time=50),
                dict(start_size=1, selfing_rate=0.3),
            ],
        )
        g = b.resolve()
        assert (
            g["a"].epochs[0].selfing_rate
            == g["b"].epochs[0].selfing_rate
            == g["c"].epochs[0].selfing_rate
            == 0.1
        )
        assert g["d"].epochs[0].selfing_rate == 0.1
        assert g["d"].epochs[1].selfing_rate == 0
        assert g["e"].epochs[0].selfing_rate == 0.2
        assert g["e"].epochs[1].selfing_rate == 0.2
        assert g["e"].epochs[2].selfing_rate == 0.3

        # cloning_rate
        b = Builder(defaults=dict(epoch=dict(cloning_rate=0.1)))
        for name in "abc":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_deme(
            "d",
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[
                dict(start_size=1, end_time=50),
                dict(start_size=1, cloning_rate=0),
            ],
        )
        b.add_deme(
            "e",
            defaults=dict(epoch=dict(cloning_rate=0.2)),
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[
                dict(start_size=1, end_time=90),
                dict(start_size=1, end_time=50),
                dict(start_size=1, cloning_rate=0.3),
            ],
        )
        g = b.resolve()
        assert (
            g["a"].epochs[0].cloning_rate
            == g["b"].epochs[0].cloning_rate
            == g["c"].epochs[0].cloning_rate
            == 0.1
        )
        assert g["d"].epochs[0].cloning_rate == 0.1
        assert g["d"].epochs[1].cloning_rate == 0
        assert g["e"].epochs[0].cloning_rate == 0.2
        assert g["e"].epochs[1].cloning_rate == 0.2
        assert g["e"].epochs[2].cloning_rate == 0.3

        # size_function
        b = Builder(defaults=dict(epoch=dict(size_function="constant")))
        for name in "abc":
            b.add_deme(name, epochs=[dict(start_size=1)])
        b.add_deme(
            "d",
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[
                dict(start_size=1, end_time=50),
                dict(start_size=1, end_size=100, size_function="exponential"),
            ],
        )
        b.add_deme(
            "e",
            defaults=dict(epoch=dict(size_function="exponential")),
            ancestors=["a", "b", "c"],
            proportions=[0.2, 0.3, 0.5],
            start_time=100,
            epochs=[
                dict(start_size=1, end_time=90, size_function="constant"),
                dict(start_size=1, end_size=100, end_time=50),
                dict(start_size=100, end_size=50, end_time=10),
                dict(start_size=100, end_size=50, size_function="N(t) = 5 * t"),
            ],
        )
        g = b.resolve()
        assert (
            g["a"].epochs[0].size_function
            == g["b"].epochs[0].size_function
            == g["c"].epochs[0].size_function
            == "constant"
        )
        assert g["d"].epochs[0].size_function == "constant"
        assert g["d"].epochs[1].size_function == "exponential"
        assert g["e"].epochs[0].size_function == "constant"
        assert g["e"].epochs[1].size_function == "exponential"
        assert g["e"].epochs[2].size_function == "exponential"
        assert g["e"].epochs[3].size_function == "N(t) = 5 * t"

    # Test demelevel epoch defaults, including overrides.
    # Comared with the test_toplevel_defaults_epoch() method, these tests
    # consider only the cases where there are no toplevel epoch defaults.
    def test_demelevel_defaults_epoch(self):
        # start_size
        b = Builder()
        b.add_deme(
            "a",
            defaults=dict(epoch=dict(start_size=1)),
        )
        b.add_deme(
            "b",
            defaults=dict(epoch=dict(start_size=1)),
            epochs=[
                dict(end_time=90),
                dict(end_size=100, end_time=50),
                dict(start_size=100, end_size=50),
            ],
        )
        g = b.resolve()
        assert g["a"].epochs[0].start_size == 1
        assert g["b"].epochs[0].start_size == 1
        assert g["b"].epochs[1].start_size == 1
        assert g["b"].epochs[2].start_size == 100

        # end_size
        b = Builder()
        b.add_deme(
            "a",
            defaults=dict(epoch=dict(end_size=1)),
        )
        b.add_deme(
            "b",
            defaults=dict(epoch=dict(end_size=1)),
            epochs=[
                dict(end_time=90),
                dict(start_size=100, end_time=50),
                dict(start_size=1, end_size=100),
            ],
        )
        g = b.resolve()
        assert g["a"].epochs[0].end_size == 1
        assert g["b"].epochs[0].end_size == 1
        assert g["b"].epochs[1].end_size == 1
        assert g["b"].epochs[2].end_size == 100

        # end_time
        b = Builder()
        b.add_deme(
            "a", defaults=dict(epoch=dict(end_time=10)), epochs=[dict(start_size=1)]
        )
        b.add_deme(
            "b",
            defaults=dict(epoch=dict(end_time=10)),  # this is silly
            epochs=[
                dict(start_size=1, end_time=90),
                dict(start_size=2, end_time=50),
                dict(start_size=3),
            ],
        )
        g = b.resolve()
        assert g["a"].end_time == 10
        assert g["a"].epochs[0].end_time == 10
        assert g["b"].end_time == 10
        assert g["b"].epochs[0].end_time == 90
        assert g["b"].epochs[1].end_time == 50
        assert g["b"].epochs[2].end_time == 10

        # selfing_rate
        b = Builder()
        b.add_deme(
            "a",
            defaults=dict(epoch=dict(selfing_rate=0.1)),
            epochs=[dict(start_size=1)],
        )
        b.add_deme(
            "b",
            defaults=dict(epoch=dict(selfing_rate=0.1)),
            epochs=[
                dict(start_size=1, end_time=90),
                dict(start_size=1, end_time=50),
                dict(start_size=1, selfing_rate=0.2),
            ],
        )
        g = b.resolve()
        assert g["a"].epochs[0].selfing_rate == 0.1
        assert g["b"].epochs[0].selfing_rate == 0.1
        assert g["b"].epochs[1].selfing_rate == 0.1
        assert g["b"].epochs[2].selfing_rate == 0.2

        # cloning_rate
        b = Builder()
        b.add_deme(
            "a",
            defaults=dict(epoch=dict(cloning_rate=0.1)),
            epochs=[dict(start_size=1)],
        )
        b.add_deme(
            "b",
            defaults=dict(epoch=dict(cloning_rate=0.1)),
            epochs=[
                dict(start_size=1, end_time=90),
                dict(start_size=1, end_time=50),
                dict(start_size=1, cloning_rate=0.2),
            ],
        )
        g = b.resolve()
        assert g["a"].epochs[0].cloning_rate == 0.1
        assert g["b"].epochs[0].cloning_rate == 0.1
        assert g["b"].epochs[1].cloning_rate == 0.1
        assert g["b"].epochs[2].cloning_rate == 0.2

        # size_function
        b = Builder()
        b.add_deme(
            "a",
            defaults=dict(epoch=dict(size_function="constant")),
            epochs=[dict(start_size=1)],
        )
        b.add_deme(
            "b",
            defaults=dict(epoch=dict(size_function="exponential")),
            epochs=[
                dict(start_size=1, end_time=90, size_function="constant"),
                dict(start_size=1, end_size=100, end_time=50),
                dict(start_size=100, end_size=50, end_time=10),
                dict(start_size=100, end_size=50, size_function="N(t) = 5 * t"),
            ],
        )
        g = b.resolve()
        assert g["a"].epochs[0].size_function == "constant"
        assert g["b"].epochs[0].size_function == "constant"
        assert g["b"].epochs[1].size_function == "exponential"
        assert g["b"].epochs[2].size_function == "exponential"
        assert g["b"].epochs[3].size_function == "N(t) = 5 * t"


class TestGraphToDict(unittest.TestCase):
    def test_finite_start_time(self):
        b = Builder()
        b.add_deme("ancestral", epochs=[dict(start_size=100)])
        b.add_deme(
            "a",
            start_time=100,
            ancestors=["ancestral"],
            epochs=[dict(start_size=100, end_time=0)],
        )
        g = b.resolve()
        d = g.asdict()
        self.assertTrue(d["demes"][1]["start_time"] == g["a"].start_time == 100)

    def test_deme_selfing_rate(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, selfing_rate=0.1)])
        d = b.resolve().asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["selfing_rate"] == 0.1)

    def test_deme_cloning_rate(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, cloning_rate=0.1)])
        d = b.resolve().asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["cloning_rate"] == 0.1)
        d = b.resolve().asdict_simplified()
        self.assertTrue(d["demes"][0]["epochs"][0]["cloning_rate"] == 0.1)

        b.add_deme("b", epochs=[dict(start_size=200, end_time=0)])
        d = b.resolve().asdict_simplified()
        self.assertTrue("cloning_rate" not in d["demes"][1])

        b.add_deme(
            "c",
            epochs=[
                dict(start_size=1, end_time=100, cloning_rate=0.3),
                dict(start_size=2),
            ],
        )
        d = b.resolve().asdict_simplified()
        self.assertTrue(d["demes"][2]["epochs"][0]["cloning_rate"] == 0.3)
        self.assertTrue("cloning_rate" not in d["demes"][2]["epochs"][1], msg=f"{d}")

    def test_fill_nonstandard_size_function(self):
        b = Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=1, end_time=10),
                dict(end_size=10, size_function="linear"),
            ],
        )
        d = b.resolve().asdict()
        self.assertTrue(d["demes"][0]["epochs"][-1]["size_function"] == "linear")

    def test_fill_epoch_selfing_rates(self):
        b = Builder()
        b.add_deme(
            "a",
            defaults={"epoch": {"selfing_rate": 0.3}},
            epochs=[
                dict(start_size=10, end_time=10, selfing_rate=0.2),
                dict(end_size=20, selfing_rate=0.1),
            ],
        )
        d = b.resolve().asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["selfing_rate"] == 0.2)
        self.assertTrue(d["demes"][0]["epochs"][1]["selfing_rate"] == 0.1)

        b = Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=10, end_time=10),
                dict(end_size=20, selfing_rate=0.1),
            ],
        )
        d = b.resolve().asdict_simplified()
        self.assertTrue("selfing_rate" not in d["demes"][0]["epochs"][0])
        self.assertTrue(d["demes"][0]["epochs"][1]["selfing_rate"] == 0.1)

    def test_fill_epoch_cloning_rates(self):
        b = Builder()
        b.add_deme(
            "a",
            defaults={"epoch": {"cloning_rate": 0.2}},
            epochs=[
                dict(start_size=10, end_time=10),
                dict(end_size=20, cloning_rate=0.1),
            ],
        )
        d = b.resolve().asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["cloning_rate"] == 0.2)
        self.assertTrue(d["demes"][0]["epochs"][1]["cloning_rate"] == 0.1)

        b = Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=10, end_time=10),
                dict(end_size=20, cloning_rate=0.1),
            ],
        )
        d = b.resolve().asdict()
        self.assertTrue(d["demes"][0]["epochs"][1]["cloning_rate"] == 0.1)

    def test_fill_description(self):
        b = Builder(description="toplevel-description")
        b.add_deme("a", description="deme-description", epochs=[dict(start_size=100)])
        g = b.resolve()
        d = g.asdict()
        self.assertTrue(d["description"] == g.description)
        self.assertTrue(d["demes"][0]["description"] == g["a"].description)

    def test_fill_migration_bounds(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, end_time=0)])
        b.add_deme("b", epochs=[dict(start_size=100, end_time=0)])
        b.add_migration(source="a", dest="b", rate=0.01, start_time=20, end_time=10)
        d = b.resolve().asdict()
        self.assertTrue(d["migrations"][0]["start_time"] == 20)
        self.assertTrue(d["migrations"][0]["end_time"] == 10)

    def test_multiple_symmetric_migrations(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=100, end_time=0)])
        b.add_deme("b", epochs=[dict(start_size=200, end_time=0)])
        b.add_deme("c", epochs=[dict(start_size=300, end_time=0)])
        b.add_migration(demes=["a", "b", "c"], rate=0.01)
        d = b.resolve().asdict_simplified()
        self.assertTrue(len(d["migrations"]) == 1)

        b.add_deme("d", epochs=[dict(start_size=400, end_time=0)])
        b.add_migration(demes=["a", "d"], rate=0.01)
        d = b.resolve().asdict_simplified()
        self.assertTrue(len(d["migrations"]) == 2)
        self.assertTrue(len(d["migrations"][0]["demes"]) == 3)
        self.assertTrue(len(d["migrations"][1]["demes"]) == 2)
        self.assertTrue("d" not in d["migrations"][0]["demes"])

    def test_mix_sym_asym_migrations(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1, end_time=0)])
        b.add_deme("b", epochs=[dict(start_size=1, end_time=0)])
        b.add_deme("c", epochs=[dict(start_size=1, end_time=0)])
        b.add_deme("d", epochs=[dict(start_size=1, end_time=0)])
        b.add_migration(demes=["a", "b"], rate=0.01)
        b.add_migration(demes=["b", "c"], rate=0.01)
        b.add_migration(demes=["a", "c", "d"], rate=0.01)
        b.add_migration(source="b", dest="d", rate=0.01)
        g = b.resolve()
        d = g.asdict()
        self.assertTrue(len(d["migrations"]) == 4)
        d = g.asdict_simplified()
        self.assertTrue(
            {"source": "b", "dest": "d", "rate": 0.01} == d["migrations"][3]
        )
        self.assertTrue(len(d["migrations"][0]["demes"]) == 2)
        self.assertTrue(len(d["migrations"][1]["demes"]) == 2)
        self.assertTrue(len(d["migrations"][2]["demes"]) == 3)

    def test_invalid_fields(self):
        b = Builder()
        b.add_deme("a", epochs=[dict(start_size=1)])
        b.data["deems"] = b.data.pop("demes")
        with pytest.raises(KeyError, match="toplevel.*deems"):
            b.resolve()

        b = Builder(defaults=dict(epok=dict(start_size=1)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError, match="defaults.*epok"):
            b.resolve()

        b = Builder(defaults=dict(epoch=dict(start_syze=1)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError, match="defaults.*epoch.*start_syze"):
            b.resolve()

        b = Builder(defaults=dict(deme=dict(end_thyme=1)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError, match="defaults.*deme.*end_thyme"):
            b.resolve()

        b = Builder(defaults=dict(migration=dict(end_thyme=1)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError, match="defaults.*migration.*end_thyme"):
            b.resolve()

        b = Builder(defaults=dict(pulse=dict(thyme=1)))
        b.add_deme("a", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError, match="defaults.*pulse.*thyme"):
            b.resolve()

        b = Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        b.add_deme("B", epochs=[dict(start_size=1)])
        b.add_deme("MacDemeFace", epochs=[dict(start_size=1)])
        b.add_deme("C", epochs=[dict(start_size=1)])
        b.data["demes"][2]["epoks"] = b.data["demes"][2].pop("epochs")
        with pytest.raises(KeyError, match="demes.*2.*MacDemeFace.*epoks"):
            b.resolve()

        b = Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        b.add_deme("B", epochs=[dict(start_size=1)])
        b.add_deme(
            "MacDemeFace",
            epochs=[
                dict(start_size=4, end_time=999),
                dict(start_size=5, end_time=99),
                dict(start_syze=99),
            ],
        )
        b.add_deme("C", epochs=[dict(start_size=1)])
        with pytest.raises(
            KeyError, match="demes.*2.*MacDemeFace.*epochs.*2.*start_syze"
        ):
            b.resolve()

        b = Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        b.add_deme("B", epochs=[dict(start_size=1)])
        b.add_deme(
            "MacDemeFace",
            defaults=dict(epok=dict(start_size=99)),
        )
        b.add_deme("C", epochs=[dict(start_size=1)])
        with pytest.raises(KeyError, match="demes.*2.*MacDemeFace.*defaults.*epok"):
            b.resolve()

        b = Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        b.add_deme("B", epochs=[dict(start_size=1)])
        b.add_deme(
            "MacDemeFace",
            defaults=dict(epoch=dict(start_syze=99)),
        )
        b.add_deme("C", epochs=[dict(start_size=1)])
        with pytest.raises(
            KeyError, match="demes.*2.*MacDemeFace.*defaults.*epoch.*start_syze"
        ):
            b.resolve()


class TestBuilder:
    def test_properties(self):
        b = Builder()
        assert hasattr(b, "data")
        assert isinstance(b.data, typing.MutableMapping)

    @hyp.settings(deadline=None, suppress_health_check=[hyp.HealthCheck.too_slow])
    @hyp.given(tests.graphs())
    def test_back_and_forth(self, graph):
        b = Builder.fromdict(graph.asdict())
        g = b.resolve()
        assert g.isclose(graph)

        b = Builder.fromdict(graph.asdict_simplified())
        g = b.resolve()
        assert g.isclose(graph)
