import io
import sys
import itertools
import argparse
from typing import Iterator, Tuple, Union
import textwrap

import demes
from . import ms


class ParseCommand:
    """
    Parse models and write them to stdout. YAML is output by default,
    but JSON or ms commands may instead be written. See options below.
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "parse",
            help="Parse models and write them to stdout in canonical form.",
            description=textwrap.dedent(self.__doc__),
        )
        parser.set_defaults(func=self)

        format_group = parser.add_mutually_exclusive_group()
        format_group.add_argument(
            "-j",
            "--json",
            action="store_true",
            default=False,
            help="Output a JSON-formatted model.",
        )
        format_group.add_argument(
            "--ms",
            metavar="REFERENCE_SIZE",
            type=float,
            default=None,
            help=(
                "Output ms command line arguments, using the given reference "
                "population size (N0) to translate into coalescent units "
                "(see the 'ms' subcommand for interpretation of this value). "
                "The sampling configuration in the output will need editing "
                "prior to simulation. The order of deme IDs matches the "
                "order of demes in the input model. "
            ),
        )

        parser.add_argument(
            "-s",
            "--simplified",
            action="store_true",
            default=False,
            help=(
                "Output a simplified model. This is a compact representation "
                "in which many default values are omitted. As only the "
                "essential details are retained, this is usually easier for "
                "humans to read. The simplified output is guaranteed to be a "
                "valid Demes model that can be resolved identically to the "
                "input model. But exactly which fields are simplified, "
                "and how simplification is performed, may change over time. "
                "Thus users should not rely on details of the output such as "
                "presence or absence of specific fields, or other details "
                "that do not alter how the model is resolved. "
                "A fully-resolved model is output by default."
            ),
        )
        parser.add_argument(
            "filename",
            type=argparse.FileType(),
            help=(
                "Filename of the model. The special value '-' may be used to "
                "read from stdin. The file may be in YAML or JSON format, "
                "but will be parsed as YAML. Multi-document YAML is supported."
            ),
        )

    def __call__(self, args: argparse.Namespace) -> None:
        if args.json:
            output_format = "json"
        elif args.ms:
            output_format = "ms"
        else:
            output_format = "yaml"

        if args.ms and args.simplified:
            # Ignore this for now.
            pass

        num_documents, graphs = self.load_and_count_documents(args.filename)
        if num_documents == 0:
            # Input file is empty.
            pass
        elif num_documents == 1:
            graph = next(graphs)
            if args.ms is not None:
                print(demes.to_ms(graph, N0=args.ms))
            else:
                demes.dump(
                    graph,
                    sys.stdout,
                    simplified=args.simplified,
                    format=output_format,
                )
        else:
            if output_format != "yaml":
                raise RuntimeError(
                    "The input file contains multiple models, which is only "
                    "supported with YAML output. If multi-model output "
                    "would be useful to you with other formats, "
                    "please open an issue on github.",
                )
            demes.dump_all(graphs, sys.stdout, simplified=args.simplified)

    def load_and_count_documents(
        self, filename: Union[str, io.TextIOBase]
    ) -> Tuple[int, Iterator[demes.Graph]]:
        """
        Count the documents in the file, returning the count and an iterator
        over the graphs. The returned document count is:
            0 for zero documents,
            1 for one document, and
            2 for two or more documents.
        """
        graph_generator = demes.load_all(filename)
        graph_list = []
        # See if we get at least two graphs. If there are more than two,
        # they'll still be lazily loaded as needed by the caller.
        for graph in graph_generator:
            graph_list.append(graph)
            if len(graph_list) > 1:
                break
        num_documents = len(graph_list)
        graph_iter = itertools.chain(graph_list, graph_generator)
        return num_documents, graph_iter


class MsCommand:
    """
    Build a Demes model from commands accepted by Hudson's classic ms simulator.
    https://doi.org/10.1093/bioinformatics/18.2.337

    Ms commands correspond to a backwards-time model of population dynamics,
    and use coalescent units for times t, population sizes x, and migration
    rates m. These are converted to more familiar units using the reference
    size N0 according to the following rules:

     - time (in generations) = 4 * N0 * t,
     - deme size (diploid individuals) = N0 * x,
     - migration rate (per generation) = m / (4 * N0).

    Deme IDs are 1-based, and migration matrix entry M[i, j] is the
    forwards-time fraction of deme i which is made up of migrants from
    deme j each generation.

    Please refer to the ms manual for the precise semantics of each command.
    http://home.uchicago.edu/~rhudson1/source/mksamples.html
    """

    def __init__(self, subparsers):
        parser = subparsers.add_parser(
            "ms",
            help="Build a Demes model using ms command line arguments.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(self.__doc__),
        )
        parser.set_defaults(func=self)

        required_group = parser.add_argument_group(
            title="required arguments", description=None
        )
        required_group.add_argument(
            "-N0",
            "--reference-size",
            type=float,
            required=True,
            help=(
                "The reference population size used to translate from coalescent "
                "units. For an existing ms command, this can be calculated as "
                "theta / (4 * mu * L), "
                "where theta is the value given to the -t option, "
                "mu is the per-generation mutation rate, "
                "and L is the length of the sequence being simulated."
            ),
        )
        ms_group = parser.add_argument_group(title="ms arguments", description=None)
        ms.build_parser(ms_group)

    def __call__(self, args: argparse.Namespace) -> None:
        graph = ms.build_graph(args, N0=args.reference_size)
        demes.dump(graph, sys.stdout)


def get_demes_parser() -> argparse.ArgumentParser:
    top_parser = argparse.ArgumentParser(
        prog="demes", description="Demes model parser and converter."
    )
    top_parser.add_argument("--version", action="version", version=demes.__version__)
    subparsers = top_parser.add_subparsers(dest="subcommand")
    ParseCommand(subparsers)
    MsCommand(subparsers)
    return top_parser


def cli(args_list=None) -> None:
    top_parser = get_demes_parser()
    args = top_parser.parse_args(args_list)
    if args.subcommand is None:
        top_parser.print_help()
        exit(1)
    args.func(args)


if __name__ == "__main__":
    cli()
