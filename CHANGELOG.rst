********************
0.1.0a3 - 2021-02-25
********************

**Bug fixes**:

- Fix ``Graph.in_generations()`` to also convert the ``Deme.start_time`` field.
  Thanks to :user:`apragsdale` for reporting the problem.
  (:user:`grahamgower`, :issue:`224`, :pr:`225`).
- Fix ``assert_close()`` and ``is_close()`` equality checks to compare the deme
  ``start_time``.
  (:user:`grahamgower`, :issue:`224`, :pr:`225`).

********************
0.1.0a2 - 2021-02-24
********************

Alpha release for testing. The API and the schema for YAML files have been
largely agreed upon. Backwards-incompatible changes before the first stable
release are still possible, but are considered unlikely.

********************
0.1.0a1 - 2020-11-12
********************

Initial alpha release to reserve the name 'demes' on pypi.
