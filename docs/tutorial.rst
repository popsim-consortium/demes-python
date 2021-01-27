.. _sec_tutorial:

========
Tutorial
========

.. _sec_tutorial_yaml_demography:

Defining a demography in YAML
-----------------------------

.. note::
   What is YAML? And why is it appropriate for this task?

A demographic model description contains enough information about
the demes (or populations) present, their attributes and relationships,
and general demographic features (e.g. rate of self-fertilization) to
be able to unambiguously recreate and simulate under that demograhy.
Thus, a ``demes`` model contains "global" attributes, specified demes, and
migration rates and mass migration events between demes.

A minimal YAML description of a demography requires the following:

#. ``description``: A description or identification of the demographic model.
#. ``time_units``: The time units for any demographic events.
#. ``demes``: a list of (at least one) deme, which must include information
   about its initial size.

For example, the simplest demography of a single population of constant
size (say, :math:`N_e=1000`) would be written as

.. literalinclude:: ./tutorial_examples/minimal.yml
   :language: yaml
   :caption: A minimal YAML demography.
   :name: minimal-demography
   :linenos:


Since we did not specify a ``start_time`` or ``end_time`` of the deme's
existence, by defult it spans all time from time :math:`\infty` in the past
to 0 ("now").  See the :ref:`next section <sec_time_units>` that describes
time conventions in a ``Graph``.

.. _sec_time_units:

Time units and conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Time flows from the past to the present. In population genetics terms, we
are in the `forward-in-time` setting (as opposed to the `backward-in-time`
setting of the coalescent). This means that when specifying migration events,
we define `source` and `destination` demes to be the demes that individuals
are moving `from` and `to`, respectively, forwards in time.

We measure time in ``time_units`` in the past, so that ``time = 0`` implies
`now` (or rather, the final generation of the simulation). Thus, time counts
down from as we advance from one generation from the next.

When defining a ``Graph`` in YAML, we must specify the ``time_units``.
Typically, we set ``time_units: generations``, so that all times are specified
in generations in the past. However, we allow the flexibility to use any unit
of time, such as years, weeks, thousands of years, etc. In the case that the
time units are not in generations, we must also specify the ``generation_time``
of the species we are modelling. For example, a human demography specified in
years could be written as

.. code-block:: yaml

    description: A human demographic model.
    time_units: years
    generation_time: 29
    demes:
      ...

Defining epochs
^^^^^^^^^^^^^^^

A ``deme`` can be made to have more interesting demographic history, such as
size changes or non-constant size functions. This is done by defining
``epochs`` that span the time that a deme exists. When defining an epoch,
we specify its ``start_time`` and ``end_time``, along with its ``start_size``
and ``end_size`` (or just ``start_size`` if the population size is constant
over that epoch).

For example, the same minimal demography :ref:`above <minimal-demography>`
could be written by specifying a single epoch over the entire time span of
the deme:

.. code-block:: yaml

   demes:
     constant_size_deme:
       epochs:
       - start_size: 1000
         end_time: 0

By default, the first listed ``epoch`` has a ``start_time`` of :math:`\infty`
if it is not specified.

To allow for size changes and varying size functions over different epochs,
we can simply specify additional epochs. Typically, we only need to define
the ``end_time`` of each epoch, as the ``start_time`` is automatically set to the
``end_time`` of the previous epoch. For this reason, ``epochs`` need to be listed
in order, from most ancient to most recent.

.. literalinclude:: ./tutorial_examples/one_pop_epochs.yml
   :language: yaml
   :caption: A single-population demography with multiple epochs.
   :name: one-pop-demography
   :linenos:

We also see, again, that for constant size epochs we only need to specify
the ``start_size``, and if no ``start_size`` is given, the ``epoch`` inherits
the ``end_size`` of the previous epoch.

In the previous example, we have a ``deme`` that expands from an effective population
size of 10,000 to 20,000 250 thousand years ago, goes through a bottleneck
between 60 and 30 thousand years ago, and then expontially grows from the
bottleneck size of 1,500 to 40,000 from 30 thousand years ago until present
time.

.. note::
   If no ``size_function`` is given, when the ``start_size`` and ``end_size``
   are different, the default size function is exponential. However, other
   size functions, such as linear, are permitted and can be specified.

Multiple demes
^^^^^^^^^^^^^^

Additional demes are specified by listing each deme under ``demes``. For example,
if we have two constant size demes that both exist for all time and never
interact (for illustration, not realism), we could write:

.. code-block:: yaml

   demes:
     deme1:
       start_size: 1000
     deme2:
       start_size: 2000

Population branches and splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's more interesting if our demes interact. First, we could have a deme that
branches off from a parental deme, and the parent continues to exist. For the child
deme, we would specify that its ``ancestor`` is its parental deme. For example,

.. code-block:: yaml

   demes:
     parental_deme:
       description: The parental deme, that exists for all time.
       start_size: 1000
     child_deme:
       description: The child deme, that exists from the branching time.
       ancestors: parental_deme
       start_size: 500
       start_time: 100

Here, the child deme split off from the parental deme 100 time units ago,
and both demes continue to exist until present time.

Alternatively, a deme could end at some time in the past, and give rise to
two or more children demes. In this case, we specify the ``end_time`` of the
parental deme, and each child deme inherits that ``end_time`` as their
``start_time``. For example, a single deme that splits into three demes
is written as

.. code-block:: yaml

   demes:
     parental_deme:
       description: The parental deme, that splits into three demes.
       start_size: 1000
       end_time: 200
     child1:
       description: The first child deme.
       ancestors: parental_deme
       start_size: 500
     child2:
       description: The second child deme.
       ancestors: parental_deme
       start_size: 800
     child3:
       description: The third child deme.
       ancestors: parental_deme
       start_size: 600

Here, the parental deme exists until time 200, and each child deme exists
from time 200 to present time.

.. literalinclude:: tutorial_examples/isolation_with_migration.yml
   :language: yaml
   :caption: A two-population isolation-with-migration model.
   :name: isolation-with-migration
   :linenos:

Continuous migration
^^^^^^^^^^^^^^^^^^^^

As suggested by the :ref:`isolation-with-migration <isolation-with-migration>`
model, continuous migration is easy to include. Continuous migration rates
are specified under ``migrations``, and can be either ``symmetric`` or
``asymmetric``. For ``symmetric`` migration, we need only specify the list of
demes involved and the migration rate. **Note that migration rates are always
given in units of per-generation**, even if the ``time_units`` are not in
generations.

For ``asymmetric`` migration, we specify the ``source`` and the ``dest`` (that is,
where migrants are coming from and going to, resp.), instead of a list of
demes. Remember that ``source`` and ``dest`` are viewed in the forward-in-time
convention.

By default, migration rates are valid for the entire interval of time that
two demes overlap. If there are periods of time that only one of the specified
demes are present, that migration rate is ingored during that time.
Optionally, we can specify a ``start_time`` and an ``end_time`` for continuous
migration, much like we specify ``epoch`` intervals.

For example, this snippet specifies two demes with changing migration rates
over time:

.. code-block:: yaml

   demes:
     deme1:
       start_size: 1000
     deme2:
       start_size: 1000
   migrations:
     symmetric:
       - demes: deme1, deme2
         rate: 1e-4
         start_time: 800
         end_time: 500
     asymmetric:
       - source: deme1
         dest: deme2
         rate: 5e-4
         start_time: 400
         end_time: 200
       - source: deme2
         dest: deme1
         rate: 2e-4
         start_time: 300
         end_time: 0

Pulse migration events
^^^^^^^^^^^^^^^^^^^^^^

Another commonly modelled process for the exchange of migrants is
a "pulse" migration event (or "mass migration" event), which is the
instantanous movement of individuals from one deme that replace
some proportion of individuals in the second deme. Such events are
specified by the ``time`` of the event, the ``source`` (where migrants
are moved from), the ``dest`` (where migrants are moved to), and the
``proportion`` of the destination deme that is replaced.

Thus, pulse migrations are specified as

.. code-block:: yaml

   pulses:
     - time: 100
       source: deme_from
       dest: deme_to
       proportion: 0.1

Of course, for this to be a valid demographic event, the ``proportion``
must be between 0 and 1, and both demes must exist at the specified
``time``.

Admixture events
^^^^^^^^^^^^^^^^

Finally, a new deme can be formed through the merger or admixture of
two or more parental demes. The parental demes could continue to exist
beyond the time of admixture, or they could each end at that time, so
that the new deme is a complete merger of its parents.
``Demes`` allows both cases - we only require that each parental deme
exists or has its ``end_time`` at the time of admixture.

To specify an admixture event, we specify the admixed deme's ``ancestors``
as a comma-separated list of parental demes, and their ``proportions`` as
a comma-separated list of admixture proportions from the parental demes.
``proportions`` must sum to 1, and the length and order must match the
demes listed in ``ancestors``.

.. code-block:: yaml

   demes:
     parental1:
       start_size: 1000
       end_time: 100
     parental2:
       start_size: 2000
       end_time: 100
     merged_deme:
       ancestors: parental1, parental2
       proportions: 0.7, 0.3
       start_size: 3000
       end_size: 5000

Here, two demes merge to form a single deme 100 time units ago, which
then grows exponentially from 3,000 to 5,000 at present time.

For a more complete example, the
:ref:`Browning et al. (2011) <browning-america>`  model of admixture
in the Americas includes three source demes that recently admix, and all
demes persist until present time.

.. literalinclude:: ./tutorial_examples/browning_america.yml
   :language: yaml
   :linenos:
   :caption: The Browning et al. (2011) American admixture model.
   :name: browning-america

.. note::
   **A note on model robustness and ambiguity**

   This is the note to be written.

Additional examples and inspiration can be found in our
:ref:`gallery of examples <sec_gallery>`.

.. _sec_tutorial_python_api:

Using the Python API
--------------------

Working directly with the Python API.

.. _sec_tutorial_attributes:

Deme attributes
---------------

Additional features: selfing, cloning, migration rate changes, ...

