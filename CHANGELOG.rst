********************
0.1.0a4 - 2021-03-22
********************

**Breaking changes**:

- The deme ``id`` field has been renamed to ``name``. This applies to both
  the data model (YAML files) and the ``Deme`` class.
  (:pr:`246`, discussion at https://github.com/popsim-consortium/demes-spec/issues/59)

**Bug fixes**:

- Check for multiple pulses causing ancestry proportion > 1.
  (:user:`grahamgower`, :issue:`250`, :pr:`251`)
- Check selfing_rate + cloning_rate <= 1.
  (:user:`grahamgower`, :issue:`242`, :pr:`251`)
- Check for pulse time edge cases.
  (:user:`grahamgower`, :issue:`243`, :pr:`249`)
- Check sum of migration rates entering a deme are <= 1.
  (:user:`grahamgower`, :issue:`244`, :pr:`249`)
- Fix migration.end_time in convert.from_msprime.
  (:user:`grahamgower`, :pr:`241`)

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
