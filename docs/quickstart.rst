.. _sec_quickstart:

==========
Quickstart
==========

Loading a Demes graph
---------------------

Consider the well-known Gutenkunst et al. (2009) Out-of-Africa model of human
history.

.. literalinclude:: ../examples/gutenkunst_ooa.yml
   :language: yaml

This YAML file can be loaded into Python with the :func:`demes.load` function,
to obtain a :class:`demes.Graph` instance (modify the filename as appropriate
for your system).

.. jupyter-execute::

    import demes

    ooa_graph = demes.load("../examples/gutenkunst_ooa.yml")
    print(isinstance(ooa_graph, demes.Graph))


Examining a Demes graph
-----------------------

The features of the graph can then be inspected. We may ask which demes are
present in the graph.

.. jupyter-execute::

    print("Is there a deme labeled CEU in the graph?", "CEU" in ooa_graph)
    print("Is there a deme labeled JPT in the graph?", "JPT" in ooa_graph)
    print("Which demes are present?", [deme.name for deme in ooa_graph.demes])

Or look in more detail at a single deme.

.. jupyter-execute::

    ceu = ooa_graph["CEU"]
    print("How many epochs does CEU have?", len(ceu.epochs))
    print(ceu.epochs[0])

Similarly, we can inspect the interactions defined between demes.

.. jupyter-execute::

    print("number of migrations:", len(ooa_graph.migrations))
    print("migrations: ")
    for migration in ooa_graph.migrations:
        print(" ", migration)

    print("number of pulses:", len(ooa_graph.pulses))


Constructing a Demes graph
--------------------------

A demographic model can instead be constructed by instantiating a
:class:`demes.Builder`, then adding demes, migrations, and admixture
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

The builder object can then be "resolved" into a :class:`demes.Graph` using the
:meth:`demes.Builder.resolve` method. We can check that our implementation
matches the example we loaded with the :meth:`demes.Graph.isclose` method.

.. jupyter-execute::

    my_graph = b.resolve()
    print(my_graph.isclose(ooa_graph))

For some demographic models, using the Python API can be far less cumbersome
than writing the equivalent YAML file. For example, we can define a ring of
demes, with migration between adjacent demes, as follows.

.. jupyter-execute::

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
    ring_graph = b.resolve()


Saving a Demes graph
--------------------

The graph can be written out to a new YAML file using :func:`demes.dump`.

.. jupyter-execute::

    demes.dump(ring_graph, "/tmp/ring.yml")
