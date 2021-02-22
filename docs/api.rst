==========
Python API
==========

Example usage
-------------

A ``yaml`` file can be loaded into python with the :func:`.load` function,
to obtain a :class:`.Graph` instance.
Here we load the Gutenkunst et al. (2009) Out-of-Africa model from
:ref:`the gallery<sec_ooa_example>`.

.. jupyter-execute::

    import demes

    g_ooa = demes.load("../examples/gutenkunst_ooa.yml")
    print("demes:", [deme.id for deme in g_ooa.demes])

    for migration in g_ooa.migrations:
        print(migration)

A demographic model can instead be constructed by instantiating a
:class:`.Builder`, then adding demes, migrations, and admixture
pulses via the methods available on this class.

.. jupyter-execute::

    b = demes.Builder(
        description="Gutenkunst et al. (2009) three-population model.",
        doi=["10.1371/journal.pgen.1000695"],
        time_units="years",
        generation_time=25,
    )
    b.add_deme("ancestral", epochs=[dict(end_time=220e3, start_size=7300)])
    b.add_deme("AMH", ancestors=["ancestral"], epochs=[dict(end_time=140e3, start_size=12300)])
    b.add_deme("OOA", ancestors=["AMH"], epochs=[dict(end_time=21.2e3, start_size=2100)])
    b.add_deme("YRI", ancestors=["AMH"], epochs=[dict(start_size=12300)])
    b.add_deme("CEU", ancestors=["OOA"], epochs=[dict(start_size=1000, end_size=29725)])
    b.add_deme("CHB", ancestors=["OOA"], epochs=[dict(start_size=510, end_size=54090)])
    b.add_migration(demes=["YRI", "OOA"], rate=25e-5)
    b.add_migration(demes=["YRI", "CEU"], rate=3e-5)
    b.add_migration(demes=["YRI", "CHB"], rate=1.9e-5)
    b.add_migration(demes=["CEU", "CHB"], rate=9.6e-5)

The builder object can then be "resolved" into a :class:`.Graph` using the
:meth:`Builder.resolve` method. We can check that our implementation matches
the example from the gallery with the :meth:`Graph.isclose` method.

.. jupyter-execute::

    graph = b.resolve()
    print(graph.isclose(g_ooa))

For some demographic models, using the Python API can be far less cumbersome
than writing the equivalent ``yaml`` file. For example, we can define a ring of
demes, with migration between adjacent demes, as follows.

.. jupyter-execute::

    import demes

    M = 10  # number of demes
    b = demes.Builder(
        description=f"a ring of {M} demes, with migration between adjacent demes",
        time_units="generations",
    )

    for j in range(M):
        b.add_deme(f"deme{j}", epochs=[dict(start_size=1000)])
        if j > 0:
            b.add_migration(demes=[f"deme{j - 1}", f"deme{j}"], rate=1e-5)
    b.add_migration(demes=[f"deme{M - 1}", "deme0"], rate=1e-5)
    graph_ring = b.resolve()

The graph can then be written out to a new ``yaml`` file using the
:func:`.dump` function.

.. jupyter-execute::

    demes.dump(graph_ring, "/tmp/ring.yml")

API Reference
-------------

.. autofunction:: demes.load
.. autofunction:: demes.loads
.. autofunction:: demes.dump
.. autofunction:: demes.dumps

.. autoclass:: demes.Builder
   :members:


.. autoclass:: demes.Graph
    :members:

.. autoclass:: demes.Deme
    :members:

.. autoclass:: demes.Epoch
    :members:

.. autoclass:: demes.Migration
    :members:

.. autoclass:: demes.AsymmetricMigration
    :members:

.. autoclass:: demes.SymmetricMigration
    :members:

.. autoclass:: demes.Pulse
    :members:

.. autoclass:: demes.Split
    :members:

.. autoclass:: demes.Branch
    :members:

.. autoclass:: demes.Merge
    :members:

.. autoclass:: demes.Admix
    :members:
