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
:class:`.Graph` directly, then adding demes, migrations, and admixture
pulses via the methods available on this class.

.. jupyter-execute::

    g = demes.Graph(
        description="Gutenkunst et al. (2009) three-population model.",
        doi=["10.1371/journal.pgen.1000695"],
        time_units="years",
        generation_time=25,
    )
    g.deme("ancestral", end_time=220e3, initial_size=7300)
    g.deme("AMH", ancestors=["ancestral"], end_time=140e3, initial_size=12300)
    g.deme("OOA", ancestors=["AMH"], end_time=21.2e3, initial_size=2100)
    g.deme("YRI", ancestors=["AMH"], initial_size=12300)
    g.deme("CEU", ancestors=["OOA"], initial_size=1000, final_size=29725)
    g.deme("CHB", ancestors=["OOA"], initial_size=510, final_size=54090)
    g.symmetric_migration(demes=["YRI", "OOA"], rate=25e-5)
    g.symmetric_migration(demes=["YRI", "CEU"], rate=3e-5)
    g.symmetric_migration(demes=["YRI", "CHB"], rate=1.9e-5)
    g.symmetric_migration(demes=["CEU", "CHB"], rate=9.6e-5)

We can check that our implementation matches the example from the gallery
with the :meth:`Graph.isclose` method.

.. jupyter-execute::

    print(g.isclose(g_ooa))

For some demographic models, using the Python API can be far less cumbersome
than writing the equivalent ``yaml`` file. For example, defining a ring of demes,
with migration between adjacent demes, can be done with the following code.

.. jupyter-execute::

    import demes

    M = 10  # number of demes
    g = demes.Graph(
        description=f"a ring of {M} demes, with migration between adjacent demes",
        time_units="generations",
    )

    for j in range(M):
        g.deme(f"deme{j}", initial_size=1000)
        if j > 0:
            g.symmetric_migration(demes=[f"deme{j - 1}", f"deme{j}"], rate=1e-5)
    g.symmetric_migration(demes=[f"deme{M - 1}", "deme0"], rate=1e-5)

The graph can then be written out to a new ``yaml`` file using the
:func:`.dump` function.

.. jupyter-execute::

    demes.dump(g, "/tmp/my_model.yml")

API Reference
-------------

.. autofunction:: demes.load
.. autofunction:: demes.dump

.. autoclass:: demes.Graph
    :members:

.. autoclass:: demes.Deme
    :members:

.. autoclass:: demes.Epoch
    :members:

.. autoclass:: demes.Migration
    :members:

.. autoclass:: demes.Pulse
    :members:
