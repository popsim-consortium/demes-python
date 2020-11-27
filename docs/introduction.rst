.. _sec_introduction:

============
Introduction
============

.. note::
   ``Demes`` is a work-in-progress, as is its documentation. If you find an error
   in the documentation or a bug in the software, or would like to help, please
   head to `Github <https://github.com/popsim-consortium/demes-python>`_ to open
   an issue or start a pull request.

Welcome to the documentation for ``Demes``, a package for defining, parsing, and
sharing demographic models for population genetic simulations.

Motivation
----------

Simulation is central to population genetics studies, and there are many great
software out there for simulating sequencing data or computing expectations of
diversity statistics under a wide range of demographic scenarios. This requires
writing a formal description of the demographic model. Generally, each
simulation software has its own syntax and style for defining the demography -
learning curves for new software can be steep and mistakes are easy to make,
especially for complex demographic scenarios.

``Demes`` aims to make defining demographic models more intuitive, less prone
to error or ambiguity, and interchangeable between simulation platforms.
Demographic models, which define populations (or `demes`), their properties, and
relationships between them, are written in `YAML <https://yaml.org/>`_. This means
that models are human-readable, and that they may then be parsed and passed to any
simulation engine that supports ``demes`` input.

For example, the following YAML file implements a simple two-epoch demographic
history for a single deme, where the deme doubles in size 100 generations ago:

.. literalinclude:: tutorial_examples/two_epoch.yaml
   :language: yaml
   :linenos:

The :ref:`tutorial <sec_tutorial_yaml_demography>` describes in detail all the
components of a YAML demographic model and how to specify more complex
scenarios, and more illustrative examples can be found in the
:ref:`Gallery <sec_gallery>`.

Getting started
---------------

- To get ``demes`` up and running, see the
  :ref:`Installation section <sec_installation>`
- In the :ref:`Tutorial section <sec_tutorial>`, you can find an introduction
  to writing demographic models using YAML, along with some bite-sized examples
  (:ref:`demographic models in YAML <sec_tutorial_yaml_demography>`), 
  manipulating and working with demography objects within python
  (:ref:`Python API <sec_tutorial_python_api>`), and adding features and
  attributes to demographic models, populations, or population epochs
  (:ref:`deme attributes <sec_tutorial_attributes>`).
- For examples of demographic models defined using ``demes``, head to the
  :ref:`Gallery <sec_gallery>`.

