import math
import pathlib
import unittest

import numpy as np
import moments

import demes
from demes import Epoch
from demes.convert import moments_


def gutenkunst_ooa():
    examples_path = pathlib.Path(__file__).parent.parent / "examples"
    return demes.load(examples_path / "gutenkunst_ooa.yml")


def moments_ooa(ns):
    fs = moments.Demographics1D.snm([sum(ns)])
    fs.integrate([1.6849315068493151], 0.2191780821917808)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1] + ns[2])
    fs.integrate(
        [1.6849315068493151, 0.2876712328767123],
        0.3254794520547945,
        m=[[0, 3.65], [3.65, 0]],
    )
    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])

    def nu_func(t):
        return [
            1.6849315068493151,
            0.136986301369863
            * np.exp(
                np.log(4.071917808219178 / 0.136986301369863) * t / 0.05808219178082192
            ),
            0.06986301369863014
            * np.exp(
                np.log(7.409589041095891 / 0.06986301369863014)
                * t
                / 0.05808219178082192
            ),
        ]

    fs.integrate(
        nu_func,
        0.05808219178082192,
        m=[[0, 0.438, 0.2774], [0.438, 0, 1.4016], [0.2774, 1.4016, 0]],
    )
    return fs


class TestMomentsSFS(unittest.TestCase):
    # test function operations
    def test_convert_to_generations(self):
        g = gutenkunst_ooa()
        sample_times = [10, 20, 50]
        sample_times_gens = [s / g.generation_time for s in sample_times]
        g, sample_times = moments_.convert_to_generations(g, sample_times)
        self.assertTrue(
            np.all([s1 == s2 for s1, s2 in zip(sample_times, sample_times_gens)])
        )

    def test_num_lineages(self):
        # simple merge model
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=100, end_time=100)])
        g.deme(
            id="pop1", ancestors=["anc"], epochs=[Epoch(initial_size=100, end_time=10)]
        )
        g.deme(
            id="pop2", ancestors=["anc"], epochs=[Epoch(initial_size=100, end_time=10)]
        )
        g.deme(
            id="pop3", ancestors=["anc"], epochs=[Epoch(initial_size=100, end_time=10)]
        )
        g.deme(
            id="pop",
            ancestors=["pop1", "pop2", "pop3"],
            proportions=[0.1, 0.2, 0.7],
            epochs=[Epoch(initial_size=100, start_time=10, end_time=0)],
        )
        sampled_demes = ["pop"]
        demes_demo_events = g.list_demographic_events()
        demo_events, demes_present = moments_.get_demographic_events(
            g, demes_demo_events, sampled_demes
        )
        deme_sample_sizes = moments_.get_deme_sample_sizes(
            g, demo_events, sampled_demes, [20], demes_present
        )
        self.assertTrue(deme_sample_sizes[(math.inf, 100)][0] == 60)
        self.assertTrue(
            np.all([deme_sample_sizes[(100, 10)][i] == 20 for i in range(3)])
        )

        # simple admix model
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=100, end_time=100)])
        g.deme(id="pop1", epochs=[Epoch(initial_size=100)], ancestors=["anc"])
        g.deme(id="pop2", epochs=[Epoch(initial_size=100)], ancestors=["anc"])
        g.deme(id="pop3", epochs=[Epoch(initial_size=100)], ancestors=["anc"])
        g.deme(
            id="pop",
            initial_size=100,
            ancestors=["pop1", "pop2", "pop3"],
            proportions=[0.1, 0.2, 0.7],
            epochs=[Epoch(initial_size=100, start_time=10, end_time=0)],
        )
        sampled_demes = ["pop"]
        demes_demo_events = g.list_demographic_events()
        demo_events, demes_present = moments_.get_demographic_events(
            g, demes_demo_events, sampled_demes
        )
        deme_sample_sizes = moments_.get_deme_sample_sizes(
            g, demo_events, sampled_demes, [20], demes_present, unsampled_n=10
        )
        self.assertTrue(deme_sample_sizes[(math.inf, 100)][0] == 90)
        self.assertTrue(
            np.all([deme_sample_sizes[(100, 10)][i] == 30 for i in range(3)])
        )

    # test basic results against moments implementation
    def test_one_pop(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Pop", epochs=[Epoch(initial_size=1000)])
        fs = moments_.SFS(g, ["Pop"], [20])
        fs_m = moments.Demographics1D.snm([20])
        self.assertTrue(np.allclose(fs.data, fs_m.data))

        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=2000),
                demes.Epoch(end_time=0, initial_size=10000),
            ],
        )
        fs = moments_.SFS(g, ["Pop"], [20])
        fs_m = moments.Demographics1D.snm([20])
        fs_m.integrate([10], 1)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_more_than_5_demes(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=1000, end_time=1000)])
        for i in range(6):
            g.deme(id=f"pop{i}", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        with self.assertRaises(ValueError):
            moments_.SFS(g, ["pop{i}" for i in range(6)], [10 for i in range(6)])

        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=1000, end_time=1000)])
        for i in range(3):
            g.deme(id=f"pop{i}", ancestors=["anc"], epochs=[Epoch(initial_size=1000)])
        with self.assertRaises(ValueError):
            moments_.SFS(
                g,
                ["pop{i}" for i in range(3)],
                [10 for i in range(3)],
                sample_times=[5, 10, 15],
            )

    def test_one_pop_ancient_samples(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Pop", epochs=[Epoch(initial_size=1000)])
        fs = moments_.SFS(g, ["Pop", "Pop"], [20, 4], sample_times=[0, 100])
        fs_m = moments.Demographics1D.snm([24])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 4)
        fs_m.integrate([1, 1], 100 / 2 / 1000, frozen=[False, True])
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_OOA(self):
        # this tests symmetric migration, size changes, splits
        g = gutenkunst_ooa()

        sample_sizes = [10, 10, 10]
        sampled_demes = ["YRI", "CEU", "CHB"]
        fs = moments_.SFS(g, sampled_demes, sample_sizes)
        # integrate with moments
        fs_m = moments_ooa(sample_sizes)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_merge(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Anc", epochs=[Epoch(initial_size=1000, end_time=100)])
        g.deme(
            id="Source1",
            ancestors=["Anc"],
            epochs=[Epoch(initial_size=2000, end_time=10)],
        )
        g.deme(
            id="Source2",
            ancestors=["Anc"],
            epochs=[Epoch(initial_size=3000, end_time=10)],
        )
        g.deme(
            id="Pop",
            ancestors=["Source1", "Source2"],
            proportions=[0.8, 0.2],
            epochs=[Epoch(initial_size=4000, start_time=10)],
        )
        fs = moments_.SFS(g, ["Pop"], [20])

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m.integrate([2, 3], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 20, 0.8)
        fs_m.integrate([4], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_admixture(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Anc", epochs=[Epoch(initial_size=1000, end_time=100)])
        g.deme(id="Source1", epochs=[Epoch(initial_size=2000)], ancestors=["Anc"])
        g.deme(id="Source2", epochs=[Epoch(initial_size=3000)], ancestors=["Anc"])
        g.deme(
            id="Pop",
            ancestors=["Source1", "Source2"],
            proportions=[0.8, 0.2],
            epochs=[Epoch(initial_size=4000, start_time=10)],
        )
        fs = moments_.SFS(g, ["Source1", "Source2", "Pop"], [10, 10, 10])

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m.integrate([2, 3], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.8)
        fs_m.integrate([2, 3, 4], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_growth_models(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(end_time=1000, initial_size=1000),
                demes.Epoch(initial_size=500, final_size=5000, end_time=0),
            ],
        )
        fs = moments_.SFS(g, ["Pop"], [100])

        fs_m = moments.Demographics1D.snm([100])

        def nu_func(t):
            return [0.5 * np.exp(np.log(5000 / 500) * t / 0.5)]

        fs_m.integrate(nu_func, 0.5)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(end_time=1000, initial_size=1000),
                demes.Epoch(
                    initial_size=500,
                    final_size=5000,
                    end_time=0,
                    size_function="linear",
                ),
            ],
        )
        fs = moments_.SFS(g, ["Pop"], [100])

        fs_m = moments.Demographics1D.snm([100])

        def nu_func(t):
            return [0.5 + t / 0.5 * (5 - 0.5)]

        fs_m.integrate(nu_func, 0.5)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_pulse_model(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=1000, end_time=100)])
        g.deme(id="source", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.deme(id="dest", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.pulse(source="source", dest="dest", time=10, proportion=0.1)
        fs = moments_.SFS(g, ["source", "dest"], [20, 20])

        fs_m = moments.Demographics1D.snm([60])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 40, 20)
        fs_m.integrate([1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_inplace(fs_m, 0, 1, 20, 0.1)
        fs_m.integrate([1, 1], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_n_way_split(self):
        g = demes.Graph(description="three-way", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=1000, end_time=10)])
        g.deme(id="deme1", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.deme(id="deme2", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.deme(id="deme3", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        ns = [10, 15, 20]
        fs = moments_.SFS(g, ["deme1", "deme2", "deme3"], ns)
        self.assertTrue(np.all([fs.sample_sizes[i] == ns[i] for i in range(len(ns))]))

        fs_m1 = moments.Demographics1D.snm([sum(ns)])
        fs_m1 = moments.Manips.split_1D_to_2D(fs_m1, ns[0], ns[1] + ns[2])
        fs_m1 = moments.Manips.split_2D_to_3D_2(fs_m1, ns[1], ns[2])
        fs_m1.integrate([1, 1, 1], 10 / 2 / 1000)

        fs_m2 = moments.Demographics1D.snm([sum(ns)])
        fs_m2 = moments.Manips.split_1D_to_2D(fs_m2, ns[0] + ns[1], ns[2])
        fs_m2 = moments.Manips.split_2D_to_3D_1(fs_m2, ns[0], ns[1])
        fs_m2 = fs_m2.swapaxes(1, 2)
        fs_m2.integrate([1, 1, 1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs.data, fs_m1.data))
        self.assertTrue(np.allclose(fs.data, fs_m2.data))

    def test_n_way_admixture(self):
        g = demes.Graph(description="three-way merge", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=1000, end_time=100)])
        g.deme(
            id="source1",
            epochs=[Epoch(initial_size=1000, end_time=10)],
            ancestors=["anc"],
        )
        g.deme(
            id="source2",
            epochs=[Epoch(initial_size=1000, end_time=10)],
            ancestors=["anc"],
        )
        g.deme(
            id="source3",
            epochs=[Epoch(initial_size=1000, end_time=10)],
            ancestors=["anc"],
        )
        g.deme(
            id="merged",
            ancestors=["source1", "source2", "source3"],
            proportions=[0.5, 0.2, 0.3],
            epochs=[Epoch(initial_size=1000, start_time=10)],
        )
        ns = [10]
        fs = moments_.SFS(g, ["merged"], ns)

        fs_m = moments.Demographics1D.snm([30])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 10, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.3)
        fs_m.integrate([1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs_m.data, fs.data))

        g = demes.Graph(description="three-way admix", time_units="generations")
        g.deme(id="anc", epochs=[Epoch(initial_size=1000, end_time=100)])
        g.deme(id="source1", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.deme(id="source2", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.deme(id="source3", epochs=[Epoch(initial_size=1000)], ancestors=["anc"])
        g.deme(
            id="admixed",
            ancestors=["source1", "source2", "source3"],
            proportions=[0.5, 0.2, 0.3],
            epochs=[Epoch(initial_size=1000, start_time=10)],
        )
        ns = [10]
        fs = moments_.SFS(g, ["admixed"], ns)

        fs_m = moments.Demographics1D.snm([30])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 10, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.3)
        fs_m.integrate([1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs_m.data[1:-1], fs.data[1:-1]))

        fs = moments_.SFS(g, ["source1", "admixed"], [10, 10])

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 1, 2, 10, 0.3)
        fs_m.integrate([1, 1], 10 / 2 / 1000)

        fs[0, 0] = fs[-1, -1] = 0
        fs_m[0, 0] = fs_m[-1, -1] = 0
        self.assertTrue(np.allclose(fs_m.data[1:-1], fs.data[1:-1]))
