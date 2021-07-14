#!/usr/bin/env python3
from setuptools import setup

setup(
    # Set the name so that github correctly tracks reverse dependencies.
    # https://github.com/popsim-consortium/demes-python/network/dependents
    name="demes",
    use_scm_version={"write_to": "demes/_version.py"},
)
