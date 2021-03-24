(sec_development)=

# Development

We welcome all contributors! Contributions to `demes` typically take the form
of "pull requests" against our [git repository][git_repo].

## Requirements

`Demes` aims to have minimal dependencies when used as a library by other
projects. However, additional dependencies are required during development, as
developers regularly run the test suite, build the documentation, and assess
whether their code changes conform to style guidelines. The `requirements.txt`
file in the top-level folder lists all development dependencies, which can
be installed using `pip`. In the following documentation, we assume the reader
has cloned the source repository and installed the developer requirements as
follows.

```sh
# Clone the repository.
git clone https://github.com/popsim-consortium/demes-python.git
cd demes
# Install the developer dependencies.
python -m pip install -r requirements.txt
# Generate the version string from the most recent git tag/commit.
python setup.py build
```

```{note}
Non-developer requirements are listed in the `install_requires` section
of the ``setup.cfg`` file in the top-level folder of the sources.
```

## Continuous integration (CI)

After a pull request is submitted, an automated process known as
*continuous integration* (CI) will:

 * assess if the proposed changes conform to style guidelines (known as *lint* checks),
 * run the test suite,
 * and build the documentation.

The CI process uses
[GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions)
and the configuration files detailing how these are run can be found under the
`.github/workflows/` folder of the sources.

## Lint checks

The following tools are run during the linting process:

 * [black](https://black.readthedocs.io/), a code formatter
   (code is only checked during CI, not reformatted),
 * [flake8](https://flake8.pycqa.org/),
   a [PEP8](https://www.python.org/dev/peps/pep-0008/) code-style checker,
 * [mypy](http://mypy-lang.org/), a static type checker.

Each of these tools can also be run manually from the top-level folder of the
sources. The `setup.cfg` file includes some project-specific configuration
for each of these tools, so running them from the command line should match
the behaviour of the CI checks.

For example, to reformat the code with `black` after making changes to the
`demes/demes.py` file (command output shown as comments):

```sh
black .
# reformatted /home/grg/src/demes/demes/demes.py
# All done! ✨ 🍰 ✨
# 1 file reformatted, 14 files left unchanged.
```

Similarly, one can check conformance to PEP8 style guidelines by running
`flake8` (without parameters), and check type annotations by running
`mypy` (also without parameters).

## Test suite

A suite of tests is included in the `tests/` folder.
The CI process uses the `pytest` tool to run the tests, which can also be run
manually from the top-level folder of the sources.

```sh
python -m pytest -v tests --cov=demes --cov-report=term-missing
```

This will produce lots of output, indicating which tests passed, and which
failed (if any). There may also be warnings. Any warnings that are triggered
by code in `demes` (rather than third-party libraries), should be fixed.

It is expected that new code contributions will be tested by the introduction
of new tests in the test suite. While we don't currently have any strict
requirements for code coverage, more is better. Furthermore, we encourage
contributions that improve, or expand on, the existing suite of tests.


## Building the documentation

The `demes` documentation is built with [jupyter-book](https://jupyter-book.org/),
which uses [sphinx](https://www.sphinx-doc.org/).
Much of the documentation is under the `docs/` folder, written in the
[MyST](https://myst-parser.readthedocs.io/en/latest/) flavour of Markdown,
and is configured in the `docs/_config.yml` file.
In contrast, the API documentation is automatically generated from "docstrings"
in the Python code that use the
[reStructuredText](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)
format. To build the documentation locally, run `make` from the `docs/` folder.

```sh
cd docs
make
```

If this was successful, the generated documentation can be viewed in a browser
by navigating to the `docs/_build/html/index.html` file. It is expected that
new code contributions will be accompanied by relevant documentation (e.g. a
new function will include a docstring).

We strongly encourage contributions that improve the `demes` documentation,
such as fixing typos and grammatical errors, or making the documentation
clearer and/or more accessible.


## Releasing a new version

```{include} release.md
```
