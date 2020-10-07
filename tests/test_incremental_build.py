import unittest

import demes

# test_examples.py tests importing and exporting YAML files
# these tests are for incremental builds of demographic models, where we
# add demes to an "empty" demography using the demographic event functions


class TestExamples(unittest.TestCase):
    maxDiff = None

    def test_one_deme_multiple_epochs(self):
        # initial size set to 50, add a later deme that has size of 100
        # only need to specify the second deme, the first one is implicit with
        # size equal to 50
        g = demes.DemeGraph(
            description="one deme test", time_units="generations", generation_time=1
        )
        g.deme(
            id="pop",
            initial_size=50,
            epochs=[demes.Epoch(start_time=30, end_time=0, initial_size=100)],
        )
        self.assertEqual(len(g["pop"].epochs), 2)
        self.assertEqual(g["pop"].epochs[0].start_time, float("inf"))
        self.assertEqual(g["pop"].epochs[0].end_time, 30)
        self.assertEqual(g["pop"].epochs[1].start_time, 30)
        self.assertEqual(g["pop"].epochs[1].end_time, 0)
        self.assertEqual(g["pop"].start_time, float("inf"))
        self.assertEqual(g["pop"].end_time, 0)

        # same as above, but start time is not inf
        g = demes.DemeGraph(
            description="one deme test", time_units="generations", generation_time=1
        )
        g.deme(
            id="pop",
            initial_size=50,
            start_time=100,
            epochs=[demes.Epoch(start_time=30, end_time=0, initial_size=100)],
        )
        self.assertEqual(len(g["pop"].epochs), 2)
        self.assertEqual(g["pop"].epochs[0].start_time, 100)

    def test_simple_split(self):
        g = demes.DemeGraph(
            description="split model", time_units="generations", generation_time=1
        )
        g.deme(id="ancestral", end_time=50, initial_size=100)
        g.deme(id="pop1", start_time=50, initial_size=200)
        g.deme(id="pop2", start_time=50, initial_size=300)
        g.split("ancestral", ["pop1", "pop2"], 50)
        self.assertEqual(len(g.splits), 1)
        self.assertTrue(g.splits[0].parent == "ancestral")
        self.assertTrue("pop1" in g.splits[0].children)
        self.assertTrue("pop2" in g.splits[0].children)
        self.assertTrue(g.splits[0].time == 50)
        self.assertTrue("ancestral" in g["pop1"].ancestors)
        self.assertTrue("ancestral" in g["pop2"].ancestors)
        self.assertTrue(g["ancestral"].ancestors is None)

    def test_simple_branch(self):
        g = demes.DemeGraph(
            description="branch model", time_units="generations", generation_time=1
        )
        g.deme(id="ancestral", initial_size=100)
        g.deme(id="pop1", start_time=50, initial_size=200)
        g.branch("ancestral", "pop1", 50)
        self.assertEqual(len(g.branches), 1)
        self.assertTrue(g.branches[0].parent == "ancestral")
        self.assertTrue(g.branches[0].child == "pop1")
        self.assertTrue(g.branches[0].time == 50)
        self.assertTrue("ancestral" in g["pop1"].ancestors)

    def test_simple_merge(self):
        g = demes.DemeGraph(
            description="branch model", time_units="generations", generation_time=1
        )
        g.deme(id="ancestral1", initial_size=100, end_time=10)
        g.deme(id="ancestral2", initial_size=100, end_time=10)
        g.deme(id="child", initial_size=100, start_time=10)
        g.merge(["ancestral1", "ancestral2"], [0.5, 0.5], "child", 10)
        self.assertEqual(len(g.mergers), 1)
        self.assertEqual(g.mergers[0].time, 10)
        self.assertEqual(g.mergers[0].child, "child")
        for anc in ["ancestral1", "ancestral2"]:
            self.assertTrue(anc in g.mergers[0].parents)
        self.assertEqual(g["ancestral1"].end_time, 10)
        self.assertEqual(g["ancestral2"].end_time, 10)
        self.assertEqual(g["child"].start_time, 10)

    def test_merge_that_truncates(self):
        # by calling merge and setting the time, we cut the parental populations
        # at the merge time
        g = demes.DemeGraph(
            description="branch model", time_units="generations", generation_time=1
        )
        g.deme(id="ancestral1", initial_size=100)  # don't set their end times
        g.deme(id="ancestral2", initial_size=100)
        g.deme(id="child", initial_size=100, start_time=10)
        g.merge(["ancestral1", "ancestral2"], [0.5, 0.5], "child", 10)
        self.assertEqual(len(g.mergers), 1)
        self.assertEqual(g.mergers[0].time, 10)
        self.assertEqual(g.mergers[0].child, "child")
        for anc in ["ancestral1", "ancestral2"]:
            self.assertTrue(anc in g.mergers[0].parents)
        self.assertEqual(g["ancestral1"].end_time, 10)
        self.assertEqual(g["ancestral2"].end_time, 10)
        self.assertEqual(g["child"].start_time, 10)

    def test_admixture(self):
        g = demes.DemeGraph(
            description="branch model", time_units="generations", generation_time=1
        )
        g.deme(id="ancestral1", initial_size=100)
        g.deme(id="ancestral2", initial_size=100)
        g.deme(id="child", initial_size=100, start_time=10)
        g.admix(["ancestral1", "ancestral2"], [0.5, 0.5], "child", 10)
        self.assertEqual(g["ancestral1"].end_time, 0)
        self.assertEqual(g["ancestral2"].end_time, 0)
        self.assertEqual(g["child"].end_time, 0)
        self.assertEqual(g["child"].start_time, 10)
        self.assertEqual(len(g.admixtures), 1)
        self.assertEqual(g.admixtures[0].time, 10)
        for anc in ["ancestral1", "ancestral2"]:
            self.assertTrue(anc in g.admixtures[0].parents)
            self.assertTrue(anc in g["child"].ancestors)
