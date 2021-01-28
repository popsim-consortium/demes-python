import unittest
import copy
import pathlib

import pytest

from demes import (
    Epoch,
    Migration,
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


class TestEpoch(unittest.TestCase):
    def test_bad_time(self):
        for time in ("0", "inf", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Epoch(start_time=time, end_time=0, start_size=1)
            with self.assertRaises(TypeError):
                Epoch(start_time=100, end_time=time, start_size=1)

        for start_time in (-10000, -1, -1e-9):
            with self.assertRaises(ValueError):
                Epoch(start_time=start_time, end_time=0, start_size=1)
        for end_time in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Epoch(start_time=100, end_time=end_time, start_size=1)

    def test_bad_time_span(self):
        with self.assertRaises(ValueError):
            Epoch(start_time=1, end_time=1, start_size=1)
        with self.assertRaises(ValueError):
            Epoch(start_time=1, end_time=2, start_size=1)

    def test_bad_size(self):
        for size in ("0", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Epoch(start_time=1, end_time=0, start_size=size)
            with self.assertRaises(TypeError):
                Epoch(start_time=1, end_time=0, start_size=1, end_size=size)

        for size in (-10000, -1, -1e-9, 0, float("inf")):
            with self.assertRaises(ValueError):
                Epoch(start_time=1, end_time=0, start_size=size)
            with self.assertRaises(ValueError):
                Epoch(start_time=1, end_time=0, start_size=1, end_size=size)

    def test_missing_size(self):
        with self.assertRaises(ValueError):
            Epoch(start_time=1, end_time=0)

    def test_valid_epochs(self):
        Epoch(end_time=0, start_size=1)
        Epoch(end_time=0, end_size=1)
        Epoch(start_time=float("inf"), end_time=0, start_size=1)
        Epoch(start_time=float("inf"), end_time=10, start_size=1)
        Epoch(start_time=100, end_time=99, start_size=1)
        Epoch(end_time=0, start_size=1, end_size=1)
        Epoch(end_time=0, start_size=1, end_size=100)
        Epoch(end_time=0, start_size=100, end_size=1)
        Epoch(start_time=20, end_time=10, start_size=1, end_size=100)
        for rate in (0, 1, 0.5, 1e-5):
            Epoch(end_time=0, start_size=1, selfing_rate=rate)
            Epoch(end_time=0, start_size=1, cloning_rate=rate)
            Epoch(end_time=0, start_size=1, selfing_rate=rate, cloning_rate=rate)
        Epoch(end_time=0, start_size=1, size_function="constant")
        Epoch(end_time=0, end_size=1, size_function="constant")
        Epoch(end_time=0, start_size=1, end_size=1, size_function="constant")
        for fn in ("linear", "exponential", "N(t) = 6 * log(t)"):
            Epoch(end_time=0, start_size=1, end_size=10, size_function=fn)
            Epoch(end_time=0, start_size=1, size_function=fn)
            Epoch(end_time=0, end_size=10, size_function=fn)

    def test_time_span(self):
        e = Epoch(start_time=float("inf"), end_time=0, start_size=1)
        self.assertEqual(e.time_span, float("inf"))
        e = Epoch(start_time=100, end_time=20, start_size=1)
        self.assertEqual(e.time_span, 80)

    def test_inf_start_time_constant_epoch(self):
        with self.assertRaises(ValueError):
            Epoch(start_time=float("inf"), end_time=0, start_size=10, end_size=20)

    def test_isclose(self):
        eps = 1e-50
        e1 = Epoch(end_time=0, start_size=1)
        self.assertTrue(e1.isclose(e1))
        self.assertTrue(e1.isclose(Epoch(end_time=0 + eps, start_size=1)))
        self.assertTrue(e1.isclose(Epoch(end_time=0, start_size=1 + eps)))

        self.assertFalse(e1.isclose(Epoch(end_time=1e-9, start_size=1)))
        self.assertFalse(e1.isclose(Epoch(end_time=0, start_size=1 + 1e-9)))
        self.assertFalse(e1.isclose(Epoch(start_time=10, end_time=0, start_size=1)))
        self.assertFalse(e1.isclose(Epoch(end_time=0, start_size=1, end_size=2)))
        self.assertFalse(
            Epoch(end_time=0, start_size=1, end_size=2).isclose(
                Epoch(
                    end_time=0,
                    start_size=1,
                    end_size=2,
                    size_function="exponential",
                )
            )
        )
        self.assertFalse(e1.isclose(Epoch(end_time=0, start_size=1, selfing_rate=0.1)))
        self.assertFalse(e1.isclose(Epoch(end_time=0, start_size=1, cloning_rate=0.1)))
        self.assertFalse(e1.isclose(None))
        self.assertFalse(e1.isclose(123))
        self.assertFalse(e1.isclose("foo"))

    def test_bad_selfing_rate(self):
        for rate in ("0", "1e-4", "inf", [], {}, float("NaN")):
            with self.assertRaises(TypeError):
                Epoch(start_time=100, end_time=0, start_size=1, selfing_rate=rate)

        for rate in (-10000, 10000, -1, -1e-9, 1.2, float("inf")):
            with self.assertRaises(ValueError):
                Epoch(start_time=100, end_time=0, start_size=1, selfing_rate=rate)

    def test_bad_cloning_rate(self):
        for rate in ("0", "1e-4", "inf", [], {}, float("NaN")):
            with self.assertRaises(TypeError):
                Epoch(start_time=100, end_time=0, start_size=1, cloning_rate=rate)

        for rate in (-10000, 10000, -1, -1e-9, 1.2, float("inf")):
            with self.assertRaises(ValueError):
                Epoch(start_time=100, end_time=0, start_size=1, cloning_rate=rate)

    def test_bad_size_function(self):
        for fn in (0, 1e5, [], {}, float("NaN")):
            with self.assertRaises(TypeError):
                Epoch(end_time=0, start_size=1, end_size=10, size_function=fn)

        for fn in ("", "constant"):
            with self.assertRaises(ValueError):
                Epoch(end_time=0, start_size=1, end_size=10, size_function=fn)


class TestMigration(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Migration(source="a", dest="b", start_time=time, end_time=0, rate=0.1)
            with self.assertRaises(TypeError):
                Migration(source="a", dest="b", start_time=100, end_time=time, rate=0.1)

        for time in (-10000, -1, -1e-9):
            with self.assertRaises(ValueError):
                Migration(source="a", dest="b", start_time=time, end_time=0, rate=0.1)
        for time in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Migration(source="a", dest="b", start_time=100, end_time=time, rate=0.1)

    def test_bad_rate(self):
        for rate in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Migration(source="a", dest="b", start_time=10, end_time=0, rate=rate)

        for rate in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Migration(source="a", dest="b", start_time=10, end_time=0, rate=rate)

    def test_bad_demes(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Migration(source=id, dest="a", start_time=10, end_time=0, rate=0.1)
            with self.assertRaises(TypeError):
                Migration(source="a", dest=id, start_time=10, end_time=0, rate=0.1)

        for id in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Migration(source=id, dest="a", start_time=10, end_time=0, rate=0.1)
            with self.assertRaises(ValueError):
                Migration(source="a", dest=id, start_time=10, end_time=0, rate=0.1)

    def test_valid_migration(self):
        Migration(source="a", dest="b", start_time=float("inf"), end_time=0, rate=1e-9)
        Migration(source="a", dest="b", start_time=1000, end_time=999, rate=0.9)

    def test_isclose(self):
        eps = 1e-50
        m1 = Migration(source="a", dest="b", start_time=1, end_time=0, rate=1e-9)
        self.assertTrue(m1.isclose(m1))
        self.assertTrue(
            m1.isclose(
                Migration(
                    source="a", dest="b", start_time=1, end_time=0, rate=1e-9 + eps
                )
            )
        )
        self.assertTrue(
            m1.isclose(
                Migration(
                    source="a", dest="b", start_time=1 + eps, end_time=0, rate=1e-9
                )
            )
        )
        self.assertTrue(
            m1.isclose(
                Migration(
                    source="a", dest="b", start_time=1, end_time=0 + eps, rate=1e-9
                )
            )
        )

        self.assertFalse(
            m1.isclose(
                Migration(source="b", dest="a", start_time=1, end_time=0, rate=1e-9)
            )
        )
        self.assertFalse(
            m1.isclose(
                Migration(source="a", dest="b", start_time=1, end_time=0, rate=2e-9)
            )
        )
        self.assertFalse(
            m1.isclose(
                Migration(source="a", dest="c", start_time=1, end_time=0, rate=1e-9)
            )
        )
        self.assertFalse(
            m1.isclose(
                Migration(source="a", dest="c", start_time=2, end_time=0, rate=1e-9)
            )
        )
        self.assertFalse(
            m1.isclose(
                Migration(source="a", dest="c", start_time=1, end_time=0.1, rate=1e-9)
            )
        )
        self.assertFalse(m1.isclose(None))
        self.assertFalse(m1.isclose(123))
        self.assertFalse(m1.isclose("foo"))


class TestPulse(unittest.TestCase):
    def test_bad_time(self):
        for time in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Pulse(source="a", dest="b", time=time, proportion=0.1)

        for time in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Pulse(source="a", dest="b", time=time, proportion=0.1)

    def test_bad_proportion(self):
        for proportion in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Pulse(source="a", dest="b", time=1, proportion=proportion)

        for proportion in (-10000, -1, -1e-9, 1.2, 100, float("inf")):
            with self.assertRaises(ValueError):
                Pulse(source="a", dest="b", time=1, proportion=proportion)

    def test_bad_demes(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Pulse(source=id, dest="a", time=1, proportion=0.1)
            with self.assertRaises(TypeError):
                Pulse(source="a", dest=id, time=1, proportion=0.1)

        for id in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Pulse(source=id, dest="a", time=1, proportion=0.1)
            with self.assertRaises(ValueError):
                Pulse(source="a", dest=id, time=1, proportion=0.1)

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
        for time in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Split(parent="a", children=["b", "c"], time=time)

        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Split(parent="a", children=["b", "c"], time=time)

    def test_bad_children(self):
        for children in (None, "b", {"b": 1}, set("b"), ("b",)):
            with self.assertRaises(TypeError):
                Split(parent="a", children=children, time=1)
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Split(parent="a", children=[id], time=1)

        for children in (["a", "b"], ["b", "b"], []):
            with self.assertRaises(ValueError):
                Split(parent="a", children=children, time=1)
        for id in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Split(parent="a", children=[id], time=1)

    def test_bad_parent(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Split(parent=id, children=["b"], time=1)

        for id in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Split(parent=id, children=["a"], time=1)

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
        for time in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Branch(parent="a", child="b", time=time)

        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Branch(parent="a", child="b", time=time)

    def test_bad_child(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Branch(parent="a", child=id, time=1)

        for id in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Branch(parent="a", child=id, time=1)

    def test_bad_parent(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Branch(parent=id, child="b", time=1)

        for id in ("a", "", "pop 1"):
            with self.assertRaises(ValueError):
                Branch(parent=id, child="a", time=1)

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
        for time in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

    def test_bad_child(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child=id, time=1)

        for id in ("a", "b", "", "pop 1"):
            with self.assertRaises(ValueError):
                Merge(parents=["a", "b"], proportions=[0.5, 0.5], child=id, time=1)

    def test_bad_parents(self):
        for parents in (None, "b", {"b": 1}, set("b"), ("b", "b")):
            with self.assertRaises(TypeError):
                Merge(parents=parents, proportions=[0.5, 0.5], child="c", time=1)
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Merge(parents=["a", id], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(TypeError):
                Merge(parents=[id, "a"], proportions=[0.5, 0.5], child="c", time=1)

        for id in ("a", "c", "", "pop 1"):
            with self.assertRaises(ValueError):
                Merge(parents=["a", id], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(ValueError):
                Merge(parents=[id, "a"], proportions=[0.5, 0.5], child="c", time=1)
        with self.assertRaises(ValueError):
            Merge(parents=["a"], proportions=[1], child="b", time=1)

    def test_bad_proportions(self):
        for proportion in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Merge(parents=["a", "b"], child="c", time=1, proportions=[proportion])

        for proportion in (-10000, -1, -1e-9, 1.2, 100, float("inf")):
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
        for time in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child="c", time=time)

    def test_bad_child(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child=id, time=1)

        for id in ("a", "b", "", "pop 1"):
            with self.assertRaises(ValueError):
                Admix(parents=["a", "b"], proportions=[0.5, 0.5], child=id, time=1)

    def test_bad_parents(self):
        for parents in (None, "b", {"b": 1}, set("b"), ("b", "b")):
            with self.assertRaises(TypeError):
                Admix(parents=parents, proportions=[0.5, 0.5], child="c", time=1)
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Admix(parents=["a", id], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(TypeError):
                Admix(parents=[id, "a"], proportions=[0.5, 0.5], child="c", time=1)

        for id in ("a", "c", "", "pop 1"):
            with self.assertRaises(ValueError):
                Admix(parents=["a", id], proportions=[0.5, 0.5], child="c", time=1)
            with self.assertRaises(ValueError):
                Admix(parents=[id, "a"], proportions=[0.5, 0.5], child="c", time=1)
        with self.assertRaises(ValueError):
            Admix(parents=["a"], proportions=[1], child="b", time=1)

    def test_bad_proportions(self):
        for proportion in ("inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Admix(parents=["a", "b"], child="c", time=1, proportions=[proportion])

        for proportion in (-10000, -1, -1e-9, 1.2, 100, float("inf")):
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
            id="a",
            description="b",
            ancestors=["c"],
            proportions=[1],
            epochs=[Epoch(start_time=float("inf"), end_time=0, start_size=1)],
            defaults={},
        )
        self.assertEqual(deme.start_time, float("inf"))
        self.assertEqual(deme.end_time, 0)
        self.assertEqual(deme.ancestors[0], "c")
        self.assertEqual(deme.proportions[0], 1)

        deme = Deme(
            id="a",
            description="b",
            ancestors=["c"],
            proportions=[1],
            epochs=[
                Epoch(start_time=100, end_time=50, start_size=1),
                Epoch(start_time=50, end_time=20, start_size=100),
                Epoch(start_time=20, end_time=1, start_size=200),
            ],
            defaults={},
        )
        self.assertEqual(deme.start_time, 100)
        self.assertEqual(deme.end_time, 1)

        deme = Deme(
            id="a",
            description=None,
            ancestors=["c"],
            proportions=[1],
            epochs=[Epoch(start_time=float("inf"), end_time=0, start_size=1)],
            defaults={},
        )
        self.assertEqual(deme.description, None)

    def test_bad_id(self):
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Deme(
                    id=id,
                    description="b",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=float("inf"), end_time=0, start_size=1)],
                    defaults={},
                )
        for id in ["", "501", "pop-1", "pop.2", "pop 3"]:
            with self.assertRaises(ValueError):
                Deme(
                    id=id,
                    description="b",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=float("inf"), end_time=0, start_size=1)],
                    defaults={},
                )

    def test_bad_description(self):
        for description in (0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Deme(
                    id="a",
                    description=description,
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=float("inf"), end_time=0, start_size=1)],
                )
        with self.assertRaises(ValueError):
            Deme(
                id="a",
                description="",
                ancestors=[],
                proportions=[],
                epochs=[Epoch(start_time=float("inf"), end_time=0, start_size=1)],
                defaults={},
            )

    def test_bad_ancestors(self):
        for ancestors in (None, "c", {}):
            with self.assertRaises(TypeError):
                Deme(
                    id="a",
                    description="b",
                    ancestors=ancestors,
                    proportions=[1],
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )
        for id in (None, 0, float("inf"), 1e3, {}, []):
            with self.assertRaises(TypeError):
                Deme(
                    id="a",
                    description="b",
                    ancestors=[id],
                    proportions=[1],
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )
        for id in ["", "501", "pop-1", "pop.2", "pop 3"]:
            with self.assertRaises(ValueError):
                Deme(
                    id="a",
                    description="b",
                    ancestors=[id],
                    proportions=[1],
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )

        with self.assertRaises(ValueError):
            Deme(
                id="a",
                description="b",
                ancestors=["a", "c"],
                proportions=[0.5, 0.5],
                epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                defaults={},
            )
        with self.assertRaises(ValueError):
            # duplicate ancestors
            Deme(
                id="a",
                description="test",
                ancestors=["x", "x"],
                proportions=[0.5, 0.5],
                epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                defaults={},
            )

    def test_bad_proportions(self):
        for proportions in (None, {}, 1e5, "proportions", float("NaN")):
            with self.assertRaises(TypeError):
                Deme(
                    id="a",
                    description="test",
                    ancestors=[],
                    proportions=proportions,
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )
        for proportion in (None, "inf", "100", {}, [], float("NaN")):
            with self.assertRaises(TypeError):
                Deme(
                    id="a",
                    description="test",
                    ancestors=["b"],
                    proportions=[proportion],
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )

        for proportions in (
            [0.6, 0.7],
            [-0.5, 1.5],
            [0, 1.0],
            [0.5, 0.2, 0.3],
        ):
            with self.assertRaises(ValueError):
                Deme(
                    id="a",
                    description="test",
                    ancestors=["x", "y"],
                    proportions=proportions,
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )

        for proportion in (-10000, -1, -1e-9, 1.2, 100, float("inf")):
            with self.assertRaises(ValueError):
                Deme(
                    id="a",
                    description="test",
                    ancestors=["b", "c"],
                    proportions=[0.5, proportion],
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )
            with self.assertRaises(ValueError):
                Deme(
                    id="a",
                    description="test",
                    ancestors=["b", "c"],
                    proportions=[proportion, 0.5],
                    epochs=[Epoch(start_time=10, end_time=0, start_size=1)],
                    defaults={},
                )

    def test_epochs_out_of_order(self):
        for time in (5, -1, float("inf")):
            with self.assertRaises(ValueError):
                Deme(
                    id="a",
                    description="b",
                    ancestors=["c"],
                    proportions=[1],
                    epochs=[
                        Epoch(start_time=10, end_time=5, start_size=1),
                        Epoch(start_time=5, end_time=time, start_size=100),
                    ],
                    defaults={},
                )

    def test_epochs_are_a_partition(self):
        for start_time, end_time in [(float("inf"), 100), (200, 100)]:
            with self.assertRaises(ValueError):
                Deme(
                    id="a",
                    description="b",
                    ancestors=["c"],
                    proportions=[1],
                    epochs=[
                        Epoch(start_time=start_time, end_time=end_time, start_size=1),
                        Epoch(start_time=50, end_time=0, start_size=2),
                    ],
                    defaults={},
                )

    def test_bad_epochs(self):
        for epochs in (None, {}, "Epoch"):
            with self.assertRaises(TypeError):
                Deme(
                    id="a",
                    description="b",
                    ancestors=["c"],
                    proportions=[1],
                    epochs=epochs,
                    defaults={},
                )
        with self.assertRaises(ValueError):
            Deme(
                id="a",
                description="b",
                ancestors=["c"],
                proportions=[1],
                epochs=[],
                defaults={},
            )

    def test_time_span(self):
        for start_time, end_time in zip((float("inf"), 100, 20), (0, 20, 0)):
            deme = Deme(
                id="a",
                description="b",
                ancestors=["c"],
                proportions=[1],
                epochs=[Epoch(start_time=start_time, end_time=end_time, start_size=1)],
                defaults={},
            )
            self.assertEqual(deme.time_span, start_time - end_time)
        with self.assertRaises(ValueError):
            deme = Deme(
                id="a",
                description="b",
                ancestors=["c"],
                proportions=[1],
                epochs=[Epoch(start_time=100, end_time=100, start_size=1)],
                defaults={},
            )

    def test_isclose(self):
        d1 = Deme(
            id="a",
            description="foo deme",
            ancestors=[],
            proportions=[],
            epochs=[Epoch(start_time=10, end_time=5, start_size=1)],
            defaults={},
        )
        self.assertTrue(d1.isclose(d1))
        self.assertTrue(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=10, end_time=5, start_size=1)],
                    defaults={},
                )
            )
        )
        # Description field doesn't matter.
        self.assertTrue(
            d1.isclose(
                Deme(
                    id="a",
                    description="bar deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=10, end_time=5, start_size=1)],
                    defaults={},
                )
            )
        )

        #
        # Check inequalities.
        #

        self.assertFalse(
            d1.isclose(
                Deme(
                    id="b",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=10, end_time=5, start_size=1)],
                    defaults={},
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=["x"],
                    proportions=[1],
                    epochs=[Epoch(start_time=10, end_time=5, start_size=1)],
                    defaults={},
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=9, end_time=5, start_size=1)],
                    defaults={},
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=10, end_time=9, start_size=1)],
                    defaults={},
                )
            )
        )

        self.assertFalse(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[Epoch(start_time=10, end_time=5, start_size=9)],
                    defaults={},
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[
                        Epoch(start_time=10, end_time=5, start_size=1, selfing_rate=0.1)
                    ],
                    defaults={},
                )
            )
        )
        self.assertFalse(
            d1.isclose(
                Deme(
                    id="a",
                    description="foo deme",
                    ancestors=[],
                    proportions=[],
                    epochs=[
                        Epoch(start_time=10, end_time=5, start_size=1, cloning_rate=0.1)
                    ],
                    defaults={},
                )
            )
        )

    # APR (7/28): Add tests for selfing rate, cloning rate, and size function.
    # Also add tests for any implied values.


class TestGraph(unittest.TestCase):
    def test_bad_generation_time(self):
        for generation_time in ([], {}, "42", "inf", float("NaN")):
            with self.assertRaises(TypeError):
                Graph(
                    description="test",
                    time_units="years",
                    generation_time=generation_time,
                )
        for generation_time in (-100, -1e-9, 0, float("inf"), None):
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
        for description in ([], {}, 0, 1e5, float("inf")):
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
        for doi_list in ({}, "10.1000/123456", float("inf"), 1e5, 0):
            with self.assertRaises(TypeError):
                Graph(
                    description="test",
                    time_units="generations",
                    doi=doi_list,
                )
        for doi in (None, {}, [], float("inf"), 1e5, 0):
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
        self.assertEqual(dg1.asdict(), dg1_copy.asdict())
        # but clearly dg2 should now differ
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

        self.assertEqual(in_generations2(dg1).asdict(), dg2.asdict())
        # in_generations2() shouldn't modify the original
        self.assertEqual(dg1.asdict(), dg1_copy.asdict())

        # in_generations() should be idempotent
        dg3 = dg2.in_generations()
        self.assertEqual(dg2.asdict(), dg3.asdict())
        dg3 = in_generations2(dg2)
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

    def test_bad_migration_time(self):
        dg = demes.Graph(description="test bad migration", time_units="generations")
        dg.deme("deme1", epochs=[Epoch(start_size=1000, end_time=0)])
        dg.deme("deme2", epochs=[Epoch(end_time=100, start_size=1000)])
        with self.assertRaises(ValueError):
            dg.migration(
                source="deme1", dest="deme2", rate=0.01, start_time=1000, end_time=0
            )

    def test_overlapping_migrations(self):
        dg = demes.Graph(description="test", time_units="generations")
        dg.deme("A", epochs=[Epoch(start_size=1, end_time=0)])
        dg.deme("B", epochs=[Epoch(start_size=1, end_time=0)])
        dg.migration(source="A", dest="B", rate=0.01)
        with self.assertRaises(ValueError):
            dg.migration(source="A", dest="B", start_time=10, rate=0.02)
        with self.assertRaises(ValueError):
            dg.symmetric_migration(demes=["A", "B"], rate=0.02)
        with self.assertRaises(ValueError):
            dg.symmetric_migration(demes=["A", "B"], rate=0.02, end_time=100)
        dg.migration(source="B", dest="A", rate=0.03, start_time=100, end_time=10)
        with self.assertRaises(ValueError):
            dg.migration(source="B", dest="A", rate=0.04, start_time=50)
        dg.migration(source="B", dest="A", rate=0.03, start_time=5)
        with self.assertRaises(ValueError):
            dg.migration(source="B", dest="A", rate=0.05, start_time=10)
        dg.migration(source="B", dest="A", rate=0.01, start_time=10, end_time=5)

    def test_bad_pulse_time(self):
        dg = demes.Graph(description="test bad pulse time", time_units="generations")
        dg.deme("deme1", epochs=[Epoch(start_size=1000, end_time=0)])
        dg.deme("deme2", epochs=[Epoch(end_time=100, start_size=1000)])
        with self.assertRaises(ValueError):
            dg.pulse(source="deme1", dest="deme2", proportion=0.1, time=10)

    def test_bad_deme(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("b", epochs=[Epoch(start_size=1, end_time=0)])
        with self.assertRaises(ValueError):
            # no epochs given
            dg.deme("a")
        with self.assertRaises(ValueError):
            # no start_size
            dg.deme("a", epochs=[Epoch(end_time=1)])
        with self.assertRaises(TypeError):
            # ancestors must be a list
            dg.deme(
                "a",
                ancestors="b",
                start_time=10,
                epochs=[Epoch(start_size=1, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # ancestor c doesn't exist
            dg.deme(
                "a",
                ancestors=["b", "c"],
                proportions=[0.5, 0.5],
                epochs=[Epoch(start_size=1, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # start_time is more recent than an epoch's start_time
            dg.deme(
                "a",
                ancestors=["b"],
                start_time=15,
                epochs=[
                    Epoch(start_size=1, start_time=20, end_time=10),
                    Epoch(end_time=0, start_size=2),
                ],
            )

    def test_duplicate_deme(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=1, end_time=0)])
        with self.assertRaises(ValueError):
            dg.deme("a", epochs=[Epoch(start_size=1, end_time=0)])

    def test_ancestor_not_in_graph(self):
        dg = demes.Graph(description="a", time_units="generations")
        with self.assertRaises(ValueError):
            dg.deme("a", ancestors=["b"], epochs=[Epoch(start_size=1)])

    def test_duplicate_ancestors(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=100, end_time=50)])
        with self.assertRaises(ValueError):
            dg.deme(
                "b",
                ancestors=["a", "a"],
                proportions=[0.5, 0.5],
                start_time=100,
                epochs=[Epoch(start_size=100, end_time=0)],
            )

    def test_bad_start_time_wrt_ancestors(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", start_time=100, epochs=[Epoch(start_size=100, end_time=50)])
        dg.deme("b", epochs=[Epoch(start_size=100, end_time=0)])
        with self.assertRaises(ValueError):
            # start_time too old
            dg.deme(
                "c",
                ancestors=["a"],
                start_time=200,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # start_time too young
            dg.deme(
                "c",
                ancestors=["a"],
                start_time=20,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # start_time too old
            dg.deme(
                "c",
                ancestors=["a", "b"],
                proportions=[0.5, 0.5],
                start_time=200,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # start_time too young
            dg.deme(
                "c",
                ancestors=["a", "b"],
                proportions=[0.5, 0.5],
                start_time=20,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # start_time not provided
            dg.deme(
                "c",
                ancestors=["a", "b"],
                proportions=[0.5, 0.5],
                epochs=[Epoch(start_size=100, end_time=0)],
            )

    def test_proportions_default(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=100, end_time=50)])
        dg.deme("b", epochs=[Epoch(start_size=100, end_time=50)])
        with self.assertRaises(ValueError):
            # proportions missing
            dg.deme(
                "c",
                ancestors=["a", "b"],
                start_time=100,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # proportions wrong length
            dg.deme(
                "c",
                ancestors=["a", "b"],
                proportions=[1],
                start_time=100,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        with self.assertRaises(ValueError):
            # proportions wrong length
            dg.deme(
                "c",
                ancestors=["a", "b"],
                proportions=[1 / 3, 1 / 3, 1 / 3],
                start_time=100,
                epochs=[Epoch(start_size=100, end_time=0)],
            )
        dg.deme("c", ancestors=["b"], epochs=[Epoch(start_size=100, end_time=0)])
        self.assertEqual(len(dg["c"].proportions), 1)
        self.assertEqual(dg["c"].proportions[0], 1.0)

    def test_bad_epochs(self):
        dg = demes.Graph(description="a", time_units="generations")
        # end_time and start_time of epochs don't align
        with self.assertRaises(ValueError):
            dg.deme(
                "a",
                epochs=[
                    Epoch(start_size=1, end_time=100),
                    Epoch(start_size=1, start_time=101, end_time=0),
                ],
            )

    def test_bad_migration(self):
        dg = demes.Graph(description="a", time_units="generations")
        with self.assertRaises(ValueError):
            dg.symmetric_migration(demes=[], rate=0)
        with self.assertRaises(ValueError):
            dg.symmetric_migration(demes=["a"], rate=0.1)
        with self.assertRaises(ValueError):
            dg.migration(source="a", dest="b", rate=0.1)
        dg.deme("a", epochs=[Epoch(start_size=100, end_time=0)])
        with self.assertRaises(ValueError):
            dg.migration(source="a", dest="b", rate=0.1)
        with self.assertRaises(ValueError):
            dg.migration(source="b", dest="a", rate=0.1)

    def test_bad_pulse(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=100, end_time=0)])
        with self.assertRaises(ValueError):
            dg.pulse(source="a", dest="b", proportion=0.1, time=10)
        with self.assertRaises(ValueError):
            dg.pulse(source="b", dest="a", proportion=0.1, time=10)

    def test_pulse_same_time(self):
        g1 = Graph(description="test", time_units="generations")
        for j in range(4):
            g1.deme(f"d{j}", epochs=[Epoch(start_size=1000, end_time=0)])

        T = 100  # time of pulses

        # Warn for duplicate pulses
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d1", time=T, proportion=0.1)
        with pytest.warns(UserWarning):
            g2.pulse(source="d0", dest="d1", time=T, proportion=0.1)

        # Warn for: d0 -> d1; d1 -> d2.
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d1", time=T, proportion=0.1)
        with pytest.warns(UserWarning):
            g2.pulse(source="d1", dest="d2", time=T, proportion=0.1)

        # Warn for: d0 -> d2; d1 -> d2.
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d2", time=T, proportion=0.1)
        with pytest.warns(UserWarning):
            g2.pulse(source="d1", dest="d2", time=T, proportion=0.1)

        # Shouldn't warn for: d0 -> d1; d0 -> d2.
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d1", time=T, proportion=0.1)
        with pytest.warns(None) as record:
            g2.pulse(source="d0", dest="d2", time=T, proportion=0.1)
        assert len(record) == 0

        # Shouldn't warn for: d0 -> d1; d2 -> d3.
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d1", time=T, proportion=0.1)
        with pytest.warns(None) as record:
            g2.pulse(source="d2", dest="d3", time=T, proportion=0.1)
        assert len(record) == 0

        # Different pulse times shouldn't warn for: d0 -> d1; d1 -> d2.
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d1", time=T, proportion=0.1)
        with pytest.warns(None) as record:
            g2.pulse(source="d1", dest="d2", time=2 * T, proportion=0.1)
        assert len(record) == 0

        # Different pulse times shouldn't warn for: d0 -> d2; d1 -> d2.
        g2 = copy.deepcopy(g1)
        g2.pulse(source="d0", dest="d2", time=T, proportion=0.1)
        with pytest.warns(None) as record:
            g2.pulse(source="d1", dest="d2", time=2 * T, proportion=0.1)
        assert len(record) == 0

    def test_isclose(self):
        g1 = Graph(
            description="test",
            time_units="generations",
        )
        g2 = copy.deepcopy(g1)
        g1.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        self.assertTrue(g1.isclose(g1))
        self.assertTrue(g1.isclose(demes.loads(demes.dumps(g1))))

        # Don't care about description for equality.
        g3 = Graph(
            description="some other description",
            time_units="generations",
        )
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        self.assertTrue(g1.isclose(g3))

        # Don't care about doi for equality.
        g3 = Graph(
            description="test",
            time_units="generations",
            doi=["https://example.com/foo.bar"],
        )
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        self.assertTrue(g1.isclose(g3))

        # The order in which demes are added shouldn't matter.
        g3 = copy.deepcopy(g2)
        g4 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        self.assertTrue(g3.isclose(g4))

        # The order in which migrations are added shouldn't matter.
        g3.migration(source="d1", dest="d2", rate=1e-4, start_time=50, end_time=40)
        g3.migration(source="d2", dest="d1", rate=1e-5)
        g4.migration(source="d2", dest="d1", rate=1e-5)
        g4.migration(source="d1", dest="d2", rate=1e-4, start_time=50, end_time=40)
        self.assertTrue(g3.isclose(g4))

        # The order in which pulses are added shouldn't matter.
        g3.pulse(source="d1", dest="d2", proportion=0.01, time=100)
        g3.pulse(source="d1", dest="d2", proportion=0.01, time=50)
        g4.pulse(source="d1", dest="d2", proportion=0.01, time=50)
        g4.pulse(source="d1", dest="d2", proportion=0.01, time=100)
        self.assertTrue(g3.isclose(g4))

        #
        # Check inequalities
        #

        self.assertFalse(g1 == g2)
        g3 = copy.deepcopy(g2)
        g3.deme("dX", epochs=[Epoch(start_size=1000, end_time=0)])
        self.assertFalse(g1.isclose(g3))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1001, end_time=0)])
        self.assertFalse(g1.isclose(g3))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        self.assertFalse(g1.isclose(g3))

        g3 = copy.deepcopy(g1)
        g4 = copy.deepcopy(g1)
        g3.deme("d2", start_time=50, epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme(
            "d2",
            ancestors=["d1"],
            start_time=50,
            epochs=[Epoch(start_size=1000, end_time=0)],
        )
        self.assertFalse(g3.isclose(g4))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4 = copy.deepcopy(g2)
        g4.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.migration(source="d2", dest="d1", rate=1e-5)
        self.assertFalse(g3.isclose(g4))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.migration(source="d1", dest="d2", rate=1e-5)
        g4 = copy.deepcopy(g2)
        g4.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.migration(source="d2", dest="d1", rate=1e-5)
        self.assertFalse(g3.isclose(g4))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.migration(source="d2", dest="d1", rate=1e-5)
        g4 = copy.deepcopy(g2)
        g4.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.symmetric_migration(demes=["d2", "d1"], rate=1e-5)
        self.assertFalse(g3.isclose(g4))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4 = copy.deepcopy(g2)
        g4.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.pulse(source="d1", dest="d2", proportion=0.01, time=100)
        self.assertFalse(g3.isclose(g4))

        g3 = copy.deepcopy(g2)
        g3.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g3.pulse(source="d2", dest="d1", proportion=0.01, time=100)
        g4 = copy.deepcopy(g2)
        g4.deme("d1", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.deme("d2", epochs=[Epoch(start_size=1000, end_time=0)])
        g4.pulse(source="d1", dest="d2", proportion=0.01, time=100)
        self.assertFalse(g3.isclose(g4))

    def test_validate(self):
        g1 = demes.Graph(description="test", time_units="generations")
        g1.deme("a", epochs=[Epoch(start_size=1, end_time=100)])
        g1.deme("b", start_time=50, epochs=[Epoch(start_size=1, end_time=0)])
        g1.validate()

        #
        # bypass the usual API to invalidate the graph
        #

        # add an ancestor deme that's not in the graph
        g2 = copy.deepcopy(g1)
        g2["a"].ancestors = ["x"]
        g2["a"].proportions = [1]
        with self.assertRaises(ValueError):
            g2.validate()

        # add an ancestor deme that's temporally not possible
        g2 = copy.deepcopy(g1)
        g2["b"].ancestors = ["a"]
        g2["b"].proportions = [1]
        with self.assertRaises(ValueError):
            g2.validate()

        # add an overlapping epoch
        g2 = copy.deepcopy(g1)
        g2["a"].epochs.append(Epoch(start_time=300, end_time=200, start_size=1))
        with self.assertRaises(ValueError):
            g2.validate()

        # add migration between non-temporally overlapping populations
        g2 = copy.deepcopy(g1)
        g2.migrations.append(
            Migration(source="a", dest="b", start_time=200, end_time=0, rate=1e-5)
        )
        with self.assertRaises(ValueError):
            g2.validate()

    def test_newly_created_objects_return(self):
        g = demes.Graph(description="test", time_units="generations")
        d1 = g.deme("a", epochs=[Epoch(start_size=1, end_time=0)])
        self.assertIsInstance(d1, Deme)
        d2 = g.deme("b", start_time=50, epochs=[Epoch(start_size=1, end_time=0)])
        self.assertIsInstance(d2, Deme)

        mig = g.migration(source="a", dest="b", rate=1e-4, end_time=25)
        self.assertIsInstance(mig, Migration)
        migs = g.symmetric_migration(demes=["a", "b"], rate=1e-4, start_time=25)
        self.assertIsInstance(migs, list)
        for m in migs:
            self.assertIsInstance(m, Migration)

        pulse = g.pulse(source="a", dest="b", proportion=0.5, time=25)
        self.assertIsInstance(pulse, Pulse)

    def test_deme_end_time(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            "a",
            epochs=[
                Epoch(end_time=100, start_size=10),
                Epoch(start_size=20, end_time=0),
            ],
        )
        with self.assertRaises(ValueError):
            g.deme("b", start_time=100, epochs=[Epoch(start_size=100, end_time=100)])
        assert "b" not in g
        with self.assertRaises(ValueError):
            g.deme("b", start_time=100, epochs=[Epoch(start_size=100, end_time=200)])
        assert "b" not in g
        # Check that end_time can be ommitted for final epoch
        g.deme("x", start_time=100, epochs=[Epoch(start_size=100)])
        g.deme("y", epochs=[Epoch(start_size=100)])
        g.deme("z", epochs=[Epoch(start_size=100, end_time=10), Epoch(start_size=10)])

    def test_ambiguous_epoch_times(self):
        g = demes.Graph(description="test", time_units="generations")
        with self.assertRaises(ValueError):
            g.deme(
                "a",
                start_time=100,
                epochs=[
                    Epoch(start_size=10),
                    Epoch(end_time=0, start_size=100),
                ],
            )


class TestGraphToDict(unittest.TestCase):
    def test_finite_start_time(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", start_time=100, epochs=[Epoch(start_size=100, end_time=0)])
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["start_time"] == dg["a"].start_time == 100)

    def test_deme_selfing_rate(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=100, selfing_rate=0.1, end_time=0)])
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["selfing_rate"] == 0.1)

    def test_deme_cloning_rate(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=100, cloning_rate=0.1, end_time=0)])
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["cloning_rate"] == 0.1)
        d = dg.asdict_simplified()
        self.assertTrue(d["demes"][0]["epochs"][0]["cloning_rate"] == 0.1)
        dg.deme("b", epochs=[Epoch(start_size=200, end_time=0)])
        d = dg.asdict_simplified()
        self.assertTrue("cloning_rate" not in d["demes"][1])
        dg.deme(
            "c",
            epochs=[
                Epoch(start_size=1, end_time=100, cloning_rate=0.3),
                Epoch(start_size=2, end_time=0),
            ],
        )
        d = dg.asdict_simplified()
        self.assertTrue(d["demes"][2]["epochs"][0]["cloning_rate"] == 0.3)
        self.assertTrue("cloning_rate" not in d["demes"][2]["epochs"][1])

    def test_fill_nonstandard_size_function(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme(
            "a",
            epochs=[
                Epoch(start_size=1, end_time=10),
                Epoch(end_size=10, size_function="linear", end_time=0),
            ],
        )
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["epochs"][-1]["size_function"] == "linear")

    def test_fill_epoch_selfing_rates(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme(
            "a",
            defaults={"selfing_rate": 0.3},
            epochs=[
                Epoch(start_size=10, end_time=10, selfing_rate=0.2),
                Epoch(end_size=20, end_time=0, selfing_rate=0.1),
            ],
        )
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["selfing_rate"] == 0.2)
        self.assertTrue(d["demes"][0]["epochs"][1]["selfing_rate"] == 0.1)

        dg = demes.Graph(description="a", time_units="generations")
        dg.deme(
            "a",
            epochs=[
                Epoch(start_size=10, end_time=10),
                Epoch(end_size=20, end_time=0, selfing_rate=0.1),
            ],
        )
        d = dg.asdict()
        self.assertTrue("selfing_rate" not in d["demes"][0]["epochs"][0])
        self.assertTrue(d["demes"][0]["epochs"][1]["selfing_rate"] == 0.1)

    def test_fill_epoch_cloning_rates(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme(
            "a",
            defaults={"cloning_rate": 0.2},
            epochs=[
                Epoch(start_size=10, end_time=10),
                Epoch(end_size=20, end_time=0, cloning_rate=0.1),
            ],
        )
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["epochs"][0]["cloning_rate"] == 0.2)
        self.assertTrue(d["demes"][0]["epochs"][1]["cloning_rate"] == 0.1)

        dg = demes.Graph(description="a", time_units="generations")
        dg.deme(
            "a",
            epochs=[
                Epoch(start_size=10, end_time=10),
                Epoch(end_size=20, end_time=0, cloning_rate=0.1),
            ],
        )
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["epochs"][1]["cloning_rate"] == 0.1)

    def test_fill_description(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme(
            "a", description="described", epochs=[Epoch(start_size=100, end_time=0)]
        )
        d = dg.asdict()
        self.assertTrue(d["demes"][0]["description"] == dg["a"].description)

    def test_fill_migration_bounds(self):
        dg = demes.Graph(description="a", time_units="generations")
        dg.deme("a", epochs=[Epoch(start_size=100, end_time=0)])
        dg.deme("b", epochs=[Epoch(start_size=100, end_time=0)])
        dg.migration(source="a", dest="b", rate=0.01, start_time=20, end_time=10)
        d = dg.asdict()
        self.assertTrue(d["migrations"]["asymmetric"][0]["start_time"] == 20)
        self.assertTrue(d["migrations"]["asymmetric"][0]["end_time"] == 10)

    def test_bad_custom_attributes(self):
        g = demes.Graph(description="a", time_units="generations")
        with self.assertRaises(TypeError):
            g.asdict_simplified(custom_attributes="inbreeding")

    def test_multiple_symmetric_migrations(self):
        g = demes.Graph(description="descr", time_units="generations")
        g.deme("a", epochs=[Epoch(start_size=100, end_time=0)])
        g.deme("b", epochs=[Epoch(start_size=200, end_time=0)])
        g.deme("c", epochs=[Epoch(start_size=300, end_time=0)])
        g.symmetric_migration(demes=["a", "b", "c"], rate=0.01)
        d = g.asdict_simplified()
        self.assertTrue(len(d["migrations"]["symmetric"]) == 1)
        self.assertTrue("asymmetric" not in d["migrations"])

        g.deme("d", epochs=[Epoch(start_size=400, end_time=0)])
        g.symmetric_migration(demes=["a", "d"], rate=0.01)
        d = g.asdict_simplified()
        self.assertTrue(len(d["migrations"]["symmetric"]) == 2)
        self.assertTrue(len(d["migrations"]["symmetric"][0]["demes"]) == 3)
        self.assertTrue(len(d["migrations"]["symmetric"][1]["demes"]) == 2)
        self.assertTrue("d" not in d["migrations"]["symmetric"][0]["demes"])
        self.assertTrue("asymmetric" not in d["migrations"])

    def test_mix_sym_asym_migrations(self):
        g = demes.Graph(description="a", time_units="generations")
        g.deme("a", start_time=100, epochs=[Epoch(start_size=1, end_time=0)])
        g.deme("b", epochs=[Epoch(start_size=1, end_time=0)])
        g.deme("c", epochs=[Epoch(start_size=1, end_time=0)])
        g.deme("d", epochs=[Epoch(start_size=1, end_time=0)])
        g.symmetric_migration(demes=["a", "b"], rate=0.01)
        g.symmetric_migration(demes=["b", "c"], rate=0.01)
        g.symmetric_migration(demes=["a", "c", "d"], rate=0.01)
        g.migration(source="b", dest="d", rate=0.01)
        d = g.asdict()
        self.assertTrue(len(d["migrations"]["asymmetric"]) == 2 + 2 + 6 + 1)
        d = g.asdict_simplified()
        self.assertTrue(
            {"source": "b", "dest": "d", "rate": 0.01} in d["migrations"]["asymmetric"]
        )
        self.assertTrue(len(d["migrations"]["symmetric"]) == 3)
        self.assertTrue(len(d["migrations"]["symmetric"][0]["demes"]) == 3)
        self.assertTrue(len(d["migrations"]["symmetric"][1]["demes"]) == 2)
        self.assertTrue(len(d["migrations"]["symmetric"][2]["demes"]) == 2)
