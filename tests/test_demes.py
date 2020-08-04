import unittest

from demes import Epoch, Migration, Pulse, Deme, DemeGraph, Split, Branch, Merge, Admix


class TestEpoch(unittest.TestCase):
    def test_bad_time(self):
        for start_time in (-10000, -1, -1e-9):
            with self.assertRaises(ValueError):
                print(start_time)
                Epoch(start_time=start_time, end_time=0, initial_size=1)
        for end_time in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Epoch(start_time=100, end_time=end_time, initial_size=1)

    def test_bad_time_span(self):
        with self.assertRaises(ValueError):
            Epoch(start_time=1, end_time=1, initial_size=1)
        with self.assertRaises(ValueError):
            Epoch(start_time=1, end_time=2, initial_size=1)

    def test_bad_size(self):
        for size in (-10000, -1, -1e-9, 0, float("inf")):
            with self.assertRaises(ValueError):
                Epoch(start_time=1, end_time=0, initial_size=size)
            with self.assertRaises(ValueError):
                Epoch(start_time=1, end_time=0, initial_size=1, final_size=size)

    def test_missing_size(self):
        with self.assertRaises(ValueError):
            Epoch(start_time=1, end_time=0)

    def test_valid_epochs(self):
        Epoch(end_time=0, initial_size=1)
        Epoch(end_time=0, final_size=1)
        Epoch(start_time=float("inf"), end_time=0, initial_size=1)
        Epoch(start_time=float("inf"), end_time=10, initial_size=1)
        Epoch(start_time=100, end_time=99, initial_size=1)
        Epoch(end_time=0, initial_size=1, final_size=1)
        Epoch(end_time=0, initial_size=1, final_size=100)
        Epoch(end_time=0, initial_size=100, final_size=1)
        Epoch(start_time=20, end_time=10, initial_size=1, final_size=100)

    def test_inf_start_time_constant_epoch(self):
        with self.assertRaises(ValueError):
            Epoch(start_time=float("inf"), end_time=0, initial_size=10, final_size=20)

    ## APR (7/28): Add tests for selfing rate, cloning rate, and size function.

class TestMigration(unittest.TestCase):
    def test_bad_time(self):
        for time in (-10000, -1, -1e-9):
            with self.assertRaises(ValueError):
                Migration("a", "b", start_time=time, end_time=0, rate=0.1)
        for time in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Migration("a", "b", start_time=100, end_time=time, rate=0.1)

    def test_bad_rate(self):
        for rate in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Migration("a", "b", start_time=10, end_time=0, rate=rate)

    def test_bad_demes(self):
        with self.assertRaises(ValueError):
            Migration("a", "a", start_time=10, end_time=0, rate=0.1)

    def test_valid_migration(self):
        Migration("a", "b", start_time=float("inf"), end_time=0, rate=1e-9)
        Migration("a", "b", start_time=1000, end_time=999, rate=0.9)


class TestPulse(unittest.TestCase):
    def test_bad_time(self):
        for time in (-10000, -1, -1e-9, float("inf")):
            with self.assertRaises(ValueError):
                Pulse("a", "b", time=time, proportion=0.1)

    def test_bad_proportion(self):
        for proportion in (-10000, -1, -1e-9, 1.2, 100, float("inf")):
            with self.assertRaises(ValueError):
                Pulse("a", "b", time=1, proportion=proportion)

    def test_bad_demes(self):
        with self.assertRaises(ValueError):
            Pulse("a", "a", time=1, proportion=0.1)

    def test_valid_pulse(self):
        Pulse("a", "b", time=1, proportion=1e-9)
        Pulse("a", "b", time=100, proportion=0.9)


class TestSplit(unittest.TestCase):
    def test_bad_time(self):
        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Split("a", ["b", "c"], time)

    def test_children(self):
        with self.assertRaises(ValueError):
            Split("a", "b", 1)
        with self.assertRaises(ValueError):
            Split("a", ["a", "b"], 1)

    def test_valid_split(self):
        Split("a", ["b", "c"], 10)
        Split("a", ["b", "c", "d"], 10)
        Split("a", ["b", "c"], 0)


class TestBranch(unittest.TestCase):
    def test_bad_time(self):
        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Branch("a", "b", time)

    def test_branch_demes(self):
        with self.assertRaises(ValueError):
            Branch("a", "a", 1)

    def test_valid_branch(self):
        Branch("a", "b", 10)
        Branch("a", "b", 0)


class TestMerge(unittest.TestCase):
    def test_bad_time(self):
        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Merge(["a", "b"], [0.5, 0.5], "c", time)

    def test_merged_demes(self):
        with self.assertRaises(ValueError):
            Merge("a", [1], "b", 1)
        with self.assertRaises(ValueError):
            Merge(["a"], [1], "b", 1)
        with self.assertRaises(ValueError):
            Merge(["a", "b"], [0.5, 0.5], "a", 1)
        with self.assertRaises(ValueError):
            Merge(["a", "a"], [0.5, 0.5], "b", 1)

    def test_invalid_proportions(self):
        with self.assertRaises(ValueError):
            Merge(["a", "b"], [0.1, 1], "c", 1)
        with self.assertRaises(ValueError):
            Merge(["a", "b"], [0.5], "c", 1)
        with self.assertRaises(ValueError):
            Merge(["a", "b"], [1.], "c", 1)
        with self.assertRaises(ValueError):
            Merge(["a", "b", "c"], [0.5, 0.5, 0.5], "d", 1)

    def test_valid_merge(self):
        Merge(["a", "b"], [0.5, 0.5], "c", 10)
        Merge(["a", "b"], [0.5, 0.5], "c", 0)
        Merge(["a", "b", "c"], [0.5, 0.25, 0.25], "d", 10)
        Merge(["a", "b", "c"], [0.5, 0.5, 0.], "d", 10)
        Merge(["a", "b"], [1, 0], "c", 10)


class TestAdmix(unittest.TestCase):
    def test_bad_time(self):
        for time in [-1e-12, -1, float("inf")]:
            with self.assertRaises(ValueError):
                Admix(["a", "b"], [0.5, 0.5], "c", time)

    def test_admixture_demes(self):
        with self.assertRaises(ValueError):
            Admix("a", [1], "b", 1)
        with self.assertRaises(ValueError):
            Admix(["a"], [1], "b", 1)
        with self.assertRaises(ValueError):
            Admix(["a", "b"], [0.5, 0.5], "a", 1)
        with self.assertRaises(ValueError):
            Admix(["a", "a"], [0.5, 0.5], "b", 1)

    def test_invalid_proportions(self):
        with self.assertRaises(ValueError):
            Admix(["a", "b"], [0.1, 1], "c", 1)
        with self.assertRaises(ValueError):
            Admix(["a", "b"], [0.5], "c", 1)
        with self.assertRaises(ValueError):
            Admix(["a", "b"], [1.], "c", 1)
        with self.assertRaises(ValueError):
            Admix(["a", "b", "c"], [0.5, 0.5, 0.5], "d", 1)

    def test_valid_admixture(self):
        Admix(["a", "b"], [0.5, 0.5], "c", 10)
        Admix(["a", "b"], [0.5, 0.5], "c", 0)
        Admix(["a", "b", "c"], [0.5, 0.25, 0.25], "d", 10)
        Admix(["a", "b", "c"], [0.5, 0.5, 0.], "d", 10)
        Admix(["a", "b"], [1, 0], "c", 10)


class TestDeme(unittest.TestCase):
    def test_properties(self):
        deme = Deme(
            "a", "b", ["c"], [1], [Epoch(start_time=float("inf"), end_time=0, initial_size=1)]
        )
        self.assertEqual(deme.start_time, float("inf"))
        self.assertEqual(deme.end_time, 0)
        self.assertEqual(deme.ancestors[0], "c")
        self.assertEqual(deme.proportions[0], 1)

        deme = Deme("a", "b", ["c"], [1], [Epoch(start_time=100, end_time=50, initial_size=1)])
        self.assertEqual(deme.start_time, 100)
        self.assertEqual(deme.end_time, 50)
        deme.add_epoch(Epoch(start_time=50, end_time=20, initial_size=100))
        self.assertEqual(deme.start_time, 100)
        self.assertEqual(deme.end_time, 20)
        deme.add_epoch(Epoch(start_time=20, end_time=1, initial_size=200))
        self.assertEqual(deme.start_time, 100)
        self.assertEqual(deme.end_time, 1)

    def test_no_epochs(self):
        with self.assertRaises(ValueError):
            Deme("a", "b", ["c"], [1], [])

    def test_two_epochs(self):
        with self.assertRaises(ValueError):
            Deme(
                "a",
                "b",
                ["c"],
                [1],
                [
                    Epoch(start_time=11, end_time=10, initial_size=100),
                    Epoch(start_time=10, end_time=1, initial_size=1),
                ],
            )

    def test_epochs_out_of_order(self):
        deme = Deme("a", "b", ["c"], [1], [Epoch(start_time=10, end_time=5, initial_size=1)])
        for time in (5, -1, float("inf")):
            with self.assertRaises(ValueError):
                deme.add_epoch(Epoch(start_time=5, end_time=time, initial_size=100))
        deme.add_epoch(Epoch(start_time=5, end_time=0, initial_size=100))

    def test_epochs_are_a_partition(self):
        for start_time, end_time in [(float("inf"), 100), (200, 100)]:
            deme = Deme(
                "a",
                "b",
                ["c"],
                [1],
                [Epoch(start_time=start_time, end_time=end_time, initial_size=1)],
            )
            with self.assertRaises(ValueError):
                deme.add_epoch(Epoch(end_time=100, initial_size=100))
            for t in (50, 20, 10):
                deme.add_epoch(Epoch(end_time=t, initial_size=t))
            prev_end_time = end_time
            for epoch in deme.epochs[1:]:
                self.assertEqual(epoch.start_time, prev_end_time)
                prev_end_time = epoch.end_time

    ## APR (7/28): Add tests for selfing rate, cloning rate, and size function.
    ## Add tests for testing ancestors and proportions.
    ## Also add tests for any implied values. 

class TestDemeGraph(unittest.TestCase):
    def test_bad_generation_time(self):
        for generation_time in (-100, -1e-9, 0, float("inf")):
            with self.assertRaises(ValueError):
                DemeGraph(
                    description="test",
                    time_units="years",
                    generation_time=generation_time,
                )

    def test_bad_default_Ne(self):
        for N in (-100, -1e-9, 0, float("inf")):
            with self.assertRaises(ValueError):
                DemeGraph(
                    description="test",
                    time_units="years",
                    generation_time=1,
                    default_Ne=N,
                )
