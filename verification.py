#!/usr/bin/env python3
# Verifies the demes.to_ms() converter by comparing summary statistics against
# msprime and moments. Tested with Python 3.9 on Linux.
import abc
import concurrent.futures
import functools
import subprocess
import os

import demes
import demesdraw
import msprime
import moments
import tsconvert
import numpy as np
import matplotlib

# Use non-GUI backend, to avoid problems with multiprocessing.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

NUM_PROCS = os.cpu_count()
NUM_REPLICATES = 100_000
REPS_PER_BATCH = 5_000
assert NUM_REPLICATES % REPS_PER_BATCH == 0
# ms isn't so useful for testing, as the trees' branch lengths are printed
# to only 3 decimal places, which is insufficient to build a tree sequence.
# We just want to check demes' ms output is implemented correctly, so mspms
# is a convenient alternative.
MS_COMMAND = "mspms"


class Simulator(abc.ABC):
    """Abstract base class for simulators."""

    def __init__(
        self, *, graph: demes.Graph, samples: dict[str, int], num_replicates=None
    ):
        self.graph = graph
        self.samples = samples
        if num_replicates is None:
            num_replicates = 1
        self.num_replicates = num_replicates

    def tmrca(self):
        """Get vector of tmrcas, one for each replicate."""
        raise NotImplementedError

    def sfs(self):
        """Get 1D SFS vector (mean or expected)."""
        raise NotImplementedError

    # Functions for simulators that output tree sequences.

    def _ts_callback(self, ts):
        """Process one simulation replicate."""
        if not hasattr(self, "_sfs"):
            self._sfs = self._ts_sfs(ts)
        else:
            self._sfs += self._ts_sfs(ts)
        if not hasattr(self, "_tmrca"):
            self._tmrca = []
        self._tmrca.append(self._ts_mean_tmrca(ts))

    def _ts_sfs(self, ts):
        """SFS branch stat."""
        return ts.allele_frequency_spectrum(mode="branch", polarised=True)

    def _ts_mean_tmrca(self, ts):
        """Mean tmrca across all trees in the sequence."""
        tmrca = []
        for tree in ts.trees():
            tmrca.append(tree.time(tree.root))
        return np.mean(tmrca)


class SimMs(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N0 = 1  # reference population size
        self.run()

    def run(self):
        samples = [self.samples.get(deme.name, 0) for deme in self.graph.demes]
        nsam = sum(samples)
        assert nsam >= 2
        # We must set a recombination rate, otherwise the output format
        # is not recognised by tsconvert.
        r = 1e-30
        sequence_length = 100
        rho = 4 * self.N0 * r * (sequence_length - 1)

        ms_args = demes.to_ms(self.graph, N0=self.N0, samples=samples)
        # We `nice` the subprocess, to avoid overcommitting the process pool
        # due to having an additional subprocess.
        cmd = (
            f"nice -n 10 {MS_COMMAND} {nsam} {self.num_replicates} {ms_args} "
            f"-T -r {rho} {sequence_length} -p 12"
        ).split()

        # Run the ms command, split the output at replicate delimiters `//`,
        # and convert each replicate into a tree sequence with tsconvert.
        num_tree_sequences = 0
        with subprocess.Popen(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            current_lines = None
            for line in process.stdout:
                line = line.rstrip()
                if line.startswith("//"):
                    # next replicate
                    if current_lines:
                        ts = tsconvert.from_ms("\n".join(current_lines))
                        self._ts_callback(ts)
                        num_tree_sequences += 1
                    current_lines = []
                elif current_lines is not None:
                    current_lines.append(line)
            stderr = process.stderr.read()

        if process.returncode != 0 or stderr.strip():
            raise RuntimeError(f"{MS_COMMAND} failed:\n" + stderr)

        if current_lines:
            ts = tsconvert.from_ms("\n".join(current_lines))
            self._ts_callback(ts)
            num_tree_sequences += 1

        assert num_tree_sequences == self.num_replicates

    def sfs(self):
        return self.N0 * np.array(self._sfs) / self.num_replicates

    def tmrca(self):
        return self.N0 * np.array(self._tmrca)


class SimMsprime(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run()

    def run(self):
        demog = msprime.Demography.from_demes(self.graph)
        ts_iter = msprime.sim_ancestry(
            demography=demog,
            samples=[
                msprime.SampleSet(nsam, population=pop, ploidy=1)
                for pop, nsam in self.samples.items()
            ],
            ploidy=2,
            num_replicates=self.num_replicates,
            record_provenance=False,
        )
        if self.num_replicates == 1:
            ts_iter = [ts_iter]

        for ts in ts_iter:
            self._ts_callback(ts)

    def sfs(self):
        return np.array(self._sfs) / 4 / self.num_replicates

    def tmrca(self):
        return np.array(self._tmrca) / 4


class SimMoments(Simulator):
    def sfs(self):
        fs = moments.Spectrum.from_demes(
            self.graph,
            sampled_demes=list(self.samples.keys()),
            sample_sizes=list(self.samples.values()),
        )
        # Scale by the ancestral size.
        # Moments only accepts graphs with one root, which is guaranteed
        # to be the first deme in the graph.
        N0 = self.graph.demes[0].epochs[0].start_size
        return fs * N0


class Parallel:
    """Wrapper that runs a simulator's replicates in parallel batches."""

    def __init__(
        self, pool, sim_class, *, num_replicates=None, reps_per_batch=None, **kwargs
    ):
        if num_replicates is None:
            num_replicates = NUM_REPLICATES
        if reps_per_batch is None:
            reps_per_batch = REPS_PER_BATCH
        # Not worth supporting non-integral multiples.
        assert num_replicates % reps_per_batch == 0
        self.futures = []
        self.num_batches = num_replicates // reps_per_batch
        for _ in range(self.num_batches):
            self.futures.append(
                pool.submit(sim_class, num_replicates=reps_per_batch, **kwargs)
            )
        self.done = False

    def _wait(self):
        sfs = None
        tmrca = []
        for fs in concurrent.futures.as_completed(self.futures):
            sim = fs.result()
            if sfs is None:
                sfs = sim.sfs()
            else:
                sfs += sim.sfs()
            tmrca.extend(sim.tmrca())
        self._sfs = sfs / self.num_batches
        self._tmrca = tmrca
        self.done = True

    def sfs(self):
        if not self.done:
            self._wait()
        return self._sfs

    def tmrca(self):
        if not self.done:
            self._wait()
        return self._tmrca


def plot_sfs(ax, title, /, **kwargs):
    """
    Plot SFS onto the given axes.
    """
    plot_styles = [
        dict(marker="o", ms=10, mfc="none", lw=2),
        dict(marker="d", mfc="none", lw=1),
        dict(marker="x", lw=1),
        dict(marker="|", lw=1),
    ]
    style = iter(plot_styles)

    for label, fs in kwargs.items():
        x = np.arange(1, len(fs) - 1, dtype=int)
        ax.plot(x, fs[1:-1], label=label, **next(style))

    ax.set_yscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Count")
    ax.set_xlabel("Allele frequency")
    ax.set_title(title)
    ax.legend()


def plot_qq(ax, title, /, **kwargs):
    """
    Plot QQ onto the given axes.
    """
    (x_label, x), (y_label, y) = kwargs.items()
    quantiles = np.linspace(0, 1, 101)
    xq = np.nanquantile(x, quantiles)
    yq = np.nanquantile(y, quantiles)
    ax.scatter(xq, yq, marker="o", edgecolor="black", facecolor="none")
    ax.scatter(xq[50], yq[50], marker="x", lw=2, c="red", label="median")

    # diagonal line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_ = min(xlim[0], ylim[0])
    max_ = max(xlim[1], ylim[1])
    ax.plot([min_, max_], [min_, max_], c="lightgray", ls="--", lw=1, zorder=-10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()


def get_axes(aspect=9 / 16, scale=1.5, **subplot_kwargs):
    """Make a matplotlib axes."""
    figsize = scale * plt.figaspect(aspect)
    fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
    fig.set_tight_layout(True)
    return fig, ax


def log_time_heuristic(graph):
    """Decide whether or not to use log time scale for demesdraw figure."""
    times = {epoch.start_time for deme in graph.demes for epoch in deme.epochs}
    times.update(epoch.end_time for deme in graph.demes for epoch in deme.epochs)
    times.discard(np.inf)
    times.discard(0)
    if len(times) > 0 and max(times) / min(times) > 4:
        log_time = True
    else:
        log_time = False
    return log_time


def multipanel_figure(pool, graph, *, sample_sets=None):
    """Multipanel figure showing the graph, TMRCA QQ, and SFS."""
    if sample_sets is None:
        nsam = 20
        sample_sets = [{deme.name: nsam} for deme in graph.demes]
    fig, axs = get_axes(nrows=2, ncols=1 + len(sample_sets))
    demesdraw.tubes(graph, ax=axs[0, 0], log_time=log_time_heuristic(graph))
    axs[1, 0].set_axis_off()

    for j, samples in enumerate(sample_sets, 1):
        ms_sims = Parallel(pool, SimMs, graph=graph, samples=samples)
        msprime_sims = Parallel(pool, SimMsprime, graph=graph, samples=samples)
        moments_sims = SimMoments(graph=graph, samples=samples)
        sample_str = ", ".join(f"{k}={v}" for k, v in samples.items())
        plot_qq(
            axs[0, j],
            f"QQ TMRCA, samples: {sample_str}",
            ms=ms_sims.tmrca(),
            msprime=msprime_sims.tmrca(),
        )
        plot_sfs(
            axs[1, j],
            f"Frequency spectrum, samples: {sample_str}",
            ms=ms_sims.sfs(),
            msprime=msprime_sims.sfs(),
            moments=moments_sims.sfs(),
        )

    return fig


def graph_zigzag():
    return demes.load("examples/zigzag.yml")


def graph_twopop_asymmetric():
    b = demes.Builder()
    b.add_deme("a", epochs=[dict(start_size=10000)])
    b.add_deme("b", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_migration(source="a", dest="b", rate=1e-3)
    return b.resolve()


def graph_twopop_symmetric():
    b = demes.Builder()
    b.add_deme("a", epochs=[dict(start_size=10000)])
    b.add_deme("b", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_migration(demes=["a", "b"], rate=1e-3)
    return b.resolve()


def graph_twopop_pulse():
    b = demes.Builder()
    b.add_deme("a", epochs=[dict(start_size=10000)])
    b.add_deme("b", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_pulse(source="a", dest="b", time=200, proportion=0.1)
    return b.resolve()


def graph_concurrent_pulses_AtoB_BtoC():
    b = demes.Builder()
    b.add_deme("a", epochs=[dict(start_size=10000)])
    b.add_deme("b", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_deme("c", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_pulse(source="a", dest="b", time=200, proportion=0.5)
    b.add_pulse(source="b", dest="c", time=200, proportion=0.5)
    return b.resolve()


def graph_concurrent_pulses_CtoB_BtoA():
    b = demes.Builder()
    b.add_deme("a", epochs=[dict(start_size=10000)])
    b.add_deme("b", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_deme("c", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_pulse(source="c", dest="b", time=200, proportion=0.5)
    b.add_pulse(source="b", dest="a", time=200, proportion=0.5)
    return b.resolve()


def graph_concurrent_pulses_AtoC_BtoC():
    b = demes.Builder()
    b.add_deme("a", epochs=[dict(start_size=10000)])
    b.add_deme("b", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_deme("c", ancestors=["a"], start_time=2000, epochs=[dict(start_size=200)])
    b.add_pulse(source="a", dest="c", time=200, proportion=0.5)
    b.add_pulse(source="b", dest="c", time=200, proportion=0.5)
    return b.resolve()


if __name__ == "__main__":
    with PdfPages("/tmp/verification.pdf") as pdf:
        with concurrent.futures.ProcessPoolExecutor(NUM_PROCS) as pool:
            fn = functools.partial(multipanel_figure, pool)
            for fig in (
                fn(graph_zigzag()),
                fn(graph_twopop_symmetric()),
                fn(graph_twopop_asymmetric()),
                fn(graph_twopop_pulse()),
                fn(
                    graph_concurrent_pulses_AtoB_BtoC(),
                    sample_sets=[dict(b=20), dict(c=20)],
                ),
                fn(
                    graph_concurrent_pulses_CtoB_BtoA(),
                    sample_sets=[dict(b=20), dict(c=20)],
                ),
                fn(
                    graph_concurrent_pulses_AtoC_BtoC(),
                    sample_sets=[dict(b=20), dict(c=20)],
                ),
            ):
                pdf.savefig(figure=fig)
                plt.close(fig)
