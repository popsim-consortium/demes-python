# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import sys

src_path = str(pathlib.Path(__file__).parent.parent)
# So we can `import demes` below.
sys.path.insert(0, src_path)
# So jupyter-sphinx finds the package.
os.environ["PYTHONPATH"] = ":".join((src_path, os.environ.get("PYTHONPATH", "")))

import demes  # noqa: E402


# -- Project information -----------------------------------------------------

project = "demes"
copyright = "2020, PopSim Consortium"
author = "PopSim Consortium"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
release = demes.__version__
# Remove trailing git hash stuff.
release = release.split("+")[0]
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_issues",
    "jupyter_sphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Github repo, for sphinx_issues
issues_github_path = "popsim-consortium/demes-python"

intersphinx_mapping = {
    "spec": ("https://popsim-consortium.github.io/demes-spec-docs/main/", None)
}
