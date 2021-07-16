import subprocess
import tempfile
import pathlib
import filecmp
import gc

import pytest

import demes
from demes.__main__ import cli


class TestTopLevel:
    def test_help(self):
        out1 = subprocess.run(
            "python -m demes -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m demes --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    def test_no_arguments_produces_help_output(self):
        # If no params are given, the output is the same as --help.
        # But the returncode should be non-zero.
        out1 = subprocess.run("python -m demes -h".split(), stdout=subprocess.PIPE)
        out2 = subprocess.run("python -m demes".split(), stdout=subprocess.PIPE)
        assert out1.stdout == out2.stdout
        assert out2.returncode != 0

    def test_version(self):
        out = subprocess.run(
            "python -m demes --version".split(),
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        assert out.stdout.strip() == demes.__version__


class TestMsCommand:
    def test_help(self):
        out1 = subprocess.run(
            "python -m demes ms -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m demes ms --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    def test_basic(self):
        b = demes.Builder()
        b.add_deme("deme1", epochs=[dict(start_size=1)])
        graph1 = b.resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "model.yaml"
            with open(tmpfile, "w") as f:
                subprocess.run("python -m demes ms -N0 1".split(), check=True, stdout=f)
            graph2 = demes.load(tmpfile)
        graph1.assert_close(graph2)

    def test_msdoc_example3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "model.yaml"
            with open(tmpfile, "w") as f:
                subprocess.run(
                    (
                        "python -m demes "
                        "ms -N0 20000 -G 6.93 -eG 0.2 0.0 -eN 0.3 0.5"
                    ).split(),
                    check=True,
                    stdout=f,
                )
            graph = demes.load(tmpfile)
        assert len(graph.migrations) == 0
        assert len(graph.pulses) == 0
        assert len(graph.demes) == 1
        assert len(graph.demes[0].epochs) == 3


class TestParseCommand:
    def test_help(self):
        out1 = subprocess.run(
            "python -m demes parse -h".split(), check=True, stdout=subprocess.PIPE
        )
        out2 = subprocess.run(
            "python -m demes parse --help".split(), check=True, stdout=subprocess.PIPE
        )
        assert out1.stdout == out2.stdout

    def test_empty_file(self):
        # An empty file should produce no output.
        out1 = subprocess.run(
            "python -m demes parse -".split(),
            check=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        assert out1.stdout == b""

    @pytest.mark.parametrize("input_format", ["json", "yaml"])
    @pytest.mark.parametrize("output_format", ["json", "yaml"])
    @pytest.mark.parametrize("simplified", [True, False])
    def test_one_graph(self, simplified, input_format, output_format):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=100)])
        graph1 = b.resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile1 = pathlib.Path(tmpdir) / "model1.yaml"
            tmpfile2 = pathlib.Path(tmpdir) / "model2.yaml"
            demes.dump(graph1, tmpfile1, format=input_format)
            with open(tmpfile2, "w") as f:
                stxt = "-s" if simplified else ""
                jtxt = "-j" if output_format == "json" else ""
                subprocess.run(
                    f"python -m demes parse {stxt} {jtxt} {tmpfile1}".split(),
                    check=True,
                    stdout=f,
                )
            graph2 = demes.load(tmpfile2, format=output_format)
            graph1.assert_close(graph2)

            demes.dump(graph1, tmpfile1, simplified=simplified, format=output_format)
            assert filecmp.cmp(tmpfile1, tmpfile2)

    @pytest.mark.parametrize("simplified", [True, False])
    def test_multiple_graphs(self, simplified):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=100)])
        graphs = [b.resolve()]
        b.add_deme("B", epochs=[dict(start_size=100)])
        graphs.append(b.resolve())
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile1 = pathlib.Path(tmpdir) / "multidoc1.yaml"
            tmpfile2 = pathlib.Path(tmpdir) / "multidoc2.yaml"
            demes.dump_all(graphs, tmpfile1)
            with open(tmpfile2, "w") as f:
                stxt = "-s" if simplified else ""
                subprocess.run(
                    f"python -m demes parse {stxt} {tmpfile1}".split(),
                    check=True,
                    stdout=f,
                )
            g1, g2 = demes.load_all(tmpfile2)
            g1.assert_close(graphs[0])
            g2.assert_close(graphs[1])

            demes.dump_all(graphs, tmpfile1, simplified=simplified)
            assert filecmp.cmp(tmpfile1, tmpfile2)

    @pytest.mark.parametrize("format_args", ["-j", "--ms 1"])
    def test_nonyaml_output_with_multiple_graphs_error(self, format_args):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=100)])
        graphs = [b.resolve()]
        b.add_deme("B", epochs=[dict(start_size=100)])
        graphs.append(b.resolve())
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile1 = pathlib.Path(tmpdir) / "multidoc1.yaml"
            demes.dump_all(graphs, tmpfile1)
            with pytest.raises(RuntimeError):
                cli(f"parse {format_args} {tmpfile1}".split())
            # Release tmpfile1. Needed on Windows for some reason.
            gc.collect()

    def test_ms_output_trivial(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        graph = b.resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "model.yaml"
            demes.dump(graph, tmpfile)
            out1 = subprocess.run(
                f"python -m demes parse --ms 1 {tmpfile}".split(),
                check=True,
                stdout=subprocess.PIPE,
                encoding="utf8",
            )
        # No output for constant size model with matching N0.
        assert out1.stdout.strip() == ""

    def test_ms_output_roundtrip(self):
        # Not all models are round-trippable, but this one should be.
        N0, N1 = 100, 200
        T0 = 300
        b = demes.Builder()
        b.add_deme(
            "deme1", epochs=[dict(start_size=N1, end_time=T0), dict(start_size=N0)]
        )
        b.add_deme(
            "deme2", ancestors=["deme1"], start_time=T0, epochs=[dict(start_size=N0)]
        )
        graph1 = b.resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = pathlib.Path(tmpdir) / "model.yaml"
            demes.dump(graph1, tmpfile)
            out = subprocess.run(
                f"python -m demes parse --ms {N0} {tmpfile}".split(),
                check=True,
                stdout=subprocess.PIPE,
                encoding="utf8",
            )
        assert out.stdout.strip() != ""
        graph2 = demes.from_ms(out.stdout, N0=N0)
        graph2.assert_close(graph1)
