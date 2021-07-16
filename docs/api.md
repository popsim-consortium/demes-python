(sec_api)=

# API Reference

## Loading and saving Demes graphs

```{eval-rst}
.. autofunction:: demes.load
.. autofunction:: demes.load_asdict
.. autofunction:: demes.loads
.. autofunction:: demes.loads_asdict
.. autofunction:: demes.load_all
.. autofunction:: demes.dump
.. autofunction:: demes.dumps
.. autofunction:: demes.dump_all
```

## Building Demes graphs

```{eval-rst}
.. autoclass:: demes.Builder
   :members:
```

## Working with Demes graphs

```{eval-rst}
.. autoclass:: demes.Graph
    :members:

    .. automethod:: __contains__

    .. automethod:: __getitem__

.. autoclass:: demes.Deme
    :members:

.. autoclass:: demes.Epoch
    :members:
```

## Continuous demographic events

```{eval-rst}
.. autoclass:: demes.AsymmetricMigration
    :members:
```

## Discrete demographic events

```{eval-rst}
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
```

## Conversion functions

```{eval-rst}
.. autofunction:: demes.from_ms
```

```{eval-rst}
.. autofunction:: demes.to_ms
```
