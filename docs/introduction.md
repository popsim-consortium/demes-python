---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]
import matplotlib.pyplot  # needed to get svg support for some reason
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")
```

(sec_introduction)=

# Introduction

The `demes` Python package provides an API for defining, parsing, and sharing
[Demes](spec:sec_intro) demographic models. Applications can use
`demes` to parse human-readable [Demes YAML files](spec:sec_tutorial)
into fully-resolved demographic models. In addition, `demes` provides
convenient data structures to simplify manipulation of demographic models.
If you find an error in the documentation or a bug in the software,
please head to our
[git repository](https://github.com/popsim-consortium/demes-python)
to report an issue or open a pull request.


## Motivation

Simulation is central to population genetics studies, and there are many great
software packages out there for simulating sequencing data or computing expectations of
diversity statistics under a wide range of demographic scenarios. This requires
writing a formal description of the demographic model. Generally, each
simulation software has its own syntax and style for defining the demography.
Learning curves for new software can be steep and mistakes are easy to make,
especially for complex demographic scenarios.

The [Demes Specification](spec:sec_intro) aims to make defining demographic
models more intuitive, less prone to error or ambiguity, and readily
interchangeable between simulation platforms.
Demographic models, which define populations (or _demes_), their properties,
and relationships between them, are by convention written as a
[YAML](https://www.yaml.info/learn/index.html) file.

## Example

The following YAML file implements a two-epoch demographic
history for a single deme, where the deme doubles in size 100 generations ago.
See the [Demes tutorial](spec:sec_tutorial) for a detailed introduction to
writing Demes YAML files.

```{literalinclude} ../examples/two_epoch.yaml
:language: yaml
```

The YAML file can be loaded using `demes`, and then visually inspected using
the [`demesdraw`](https://github.com/grahamgower/demesdraw) Python package.

```{code-cell}
import demes
import demesdraw

graph = demes.load("../examples/two_epoch.yaml")
demesdraw.tubes(graph)
```
