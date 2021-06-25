import sys
import functools

import click
import ruamel

import demes


@click.group()
def main():
    pass


@main.command(context_settings={"ignore_unknown_options": True})
@click.option("-N0", "N0", type=float, required=True, help="Reference population size")
@click.argument("commands", required=True, nargs=-1)
def ms(N0, commands):
    """
    Convert an ms command to a Demes model. E.g. (from the ms manual):

        demes ms 15 1000 -t 6.4 -G 6.93 -eG 0.2 0.0 -eN 0.3 0.5 -N0 20000
    """
    graph = demes.from_ms(" ".join(commands), N0=N0)
    demes.dump(graph, sys.stdout)


def detect_input_format(input_format, filename):
    if input_format is None:
        if filename.endswith("json"):
            input_format = "json"
        else:
            input_format = "yaml"
    return input_format


@main.command()
@click.option(
    "-f",
    "--fully-resolved",
    is_flag=True,
    default=False,
    help="Output fully-resolved model. A simplified model is output by default.",
)
@click.option("-i", "--input-format", type=click.Choice(["yaml", "json"]), default=None)
@click.option(
    "-o", "--output-format", type=click.Choice(["yaml", "json"]), default="yaml"
)
@click.argument("filename", type=str, default="-")
def convert(fully_resolved: bool, input_format: str, output_format: str, filename):
    """
    Convert between JSON/YAML formats or simplified/fully-qualified variants.
    """
    input_format = detect_input_format(input_format, filename)
    if filename == "-":
        filename = sys.stdin
    dump_func = functools.partial(demes.dump, format=output_format)
    multidoc = False
    try:
        # Try single document loader.
        graph_arg = demes.load(filename, format=input_format)
    except ruamel.yaml.composer.ComposerError:
        # ruamel complained about multiple documents
        multidoc = True

    if multidoc:
        assert input_format == "yaml"
        if output_format != "yaml":
            print("multi-document files only supported with --output-format yaml")
            exit(1)
        graph_arg = demes.load_all(filename)
        dump_func = demes.dump_all

    dump_func(
        graph_arg,
        sys.stdout,
        simplified=not fully_resolved,
    )


@main.command()
@click.option("-i", "--input-format", type=click.Choice(["yaml", "json"]), default=None)
@click.argument("filename", default="-")
def validate(input_format: str, filename):
    """
    Validate a model.
    """
    input_format = detect_input_format(input_format, filename)
    if filename == "-":
        filename = sys.stdin

    multidoc = False
    try:
        # Try single document loader.
        demes.load(filename, format=input_format)
    except ruamel.yaml.composer.ComposerError:
        # ruamel complained about multiple documents
        multidoc = True

    if multidoc:
        assert input_format == "yaml"
        for _ in demes.load_all(filename):
            pass


@main.command()
@click.option("-i", "--input-format", type=click.Choice(["yaml", "json"]), default=None)
@click.option("-O", "--output-file", type=str, default=None)
@click.argument("filename", default="-")
def plot(input_format: str, output_file, filename):
    """
    Plot a model using the demesdraw library.
    """
    try:
        import demesdraw
    except ImportError:
        print("must install demesdraw to use the plot subcommand")
        exit(1)

    input_format = detect_input_format(input_format, filename)
    if filename == "-":
        filename = sys.stdin
    graph = demes.load(filename, format=input_format)
    ax = demesdraw.tubes(graph)
    if output_file is not None:
        ax.figure.savefig(output_file)
    else:
        # interactive plot
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
