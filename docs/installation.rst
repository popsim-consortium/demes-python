.. _sec_installation:

============
Installation
============

Requirements
------------

``Demes`` uses `ruamel.yaml <https://pypi.org/project/ruamel.yaml/>`_ to
read YAML-defined demographic models.

Installation
------------

Currently, ``demes`` is most useful for simulation software developers who
want to support demographic models imported from ``demes`` descriptions.

Installation instructions for developers:

.. code-block:: sh

   ## Clone the repository
   git clone https://github.com/popsim-consortium/demes-python.git
   cd demes
   
   ## Install the developer dependencies
   python -m pip install -r requirements.txt
   
   ## Run the test suite
   python -m pytest -v tests


