# Changelog

## 0.2.3 - 2023-05-04

- Fixed the use of numpy strings for deme names.
  ({user}`terhorst`, {user}`grahamgower`, {issue}`495`, {pr}`505`)
- Added `Graph.rename_demes()` method to return a graph with renamed demes.
  ({user}`aabiddanda`, {pr}`499`)

## 0.2.2 - 2022-04-28

Better conformance to the spec. Minor discrepancies between
demes-python and the reference implementation have been resolved.

- Dropped support for Python 3.6
  ({user}`grahamgower`, {issue}`445`, {pr}`446`)
- Reject bad defaults, even if they're not used.
  ({user}`grahamgower`, {issue}`443`, {pr}`430`)
- Set encoding to UTF-8 explicitly. This fixes unicode deme names on Windows.
  ({user}`grahamgower`, {pr}`430`)
- `Graph.description` and `Deme.description` can no longer be `None`.
  A missing description will now be resolved to the empty string.
  ({user}`grahamgower`, {issue}`429`, {pr}`430`)
- `Graph.generation_time` can no longer be `None`.
  When `time_units` are generations, a missing `generation_time` will now
  be resolved to 1 and other values are an error.
  ({user}`grahamgower`, {issue}`429`, {pr}`430`)
- Permit selfing_rate + cloning_rate >= 1.
  Both values must be between zero and one (inclusive), but the selfing_rate
  is now defined as conditional on sexual reproduction (which occurs at
  rate `1 - cloning_rate`).
  ({user}`grahamgower`, {issue}`425`, {pr}`423`)

## 0.2.1 - 2021-12-07

**New features:**

- Support for the `metadata` field at the toplevel of a YAML file.
  The is a dictionary that may contain arbitrary nested data.
  ({user}`grahamgower`, {issue}`275`, {pr}`392`).

**Bug fixes:**

- The string "Infinity" is now accepted when using `load_all()`,
  just like for `load()` and `loads()`.
  This fixes loading fully-resolved models with the CLI.
  ({user}`grahamgower`, {issue}`394`, {pr}`395`).

**Breaking changes:**

- The `demes.hypothesis_strategies.graphs()` function for generating
  a random `Graph` has been removed. This was buggy and not usable
  as originally intended.
  ({user}`grahamgower`, {issue}`360`, {pr}`397`).

## 0.2.0 - 2021-12-01

**New features:**

- Add `load_all()`/`dump_all()` to support multi-document YAML.
  ({user}`grahamgower`, {issue}`239`, {pr}`335`)
- Add CLI.
  ({user}`grahamgower`, {pr}`339`)
- Add `to_ms()` function.
  ({user}`grahamgower`, {issue}`74`, {pr}`354`)
- Allow for "Infinity" as start times for demes and migrations in input
  YAMLs and dicts.
  ({user}`apragsdale`, {issue}`358`, {pr}`386`)


**Breaking changes:**

- A pulse event now allows for simultaneous sources and takes arguments
  `sources` and `proportions` instead of `source` and `proportion`. The
  sources and proportions must be provided as a list, even when there is
  only a single source deme.
  ({user}`apragsdale`, {pr}`353`)
- Disallow null values and blank entries in input YAML models.
  ({user}`apragsdale`, {issue}`340`, {pr}`387`)

**Bug fixes:**

- Fix various issues in `from_ms()` with `-es`/`-ej` commands.
  ({user}`grahamgower`, {issue}`350`, {issue}`351`, {pr}`352`)
- Fix some pulse edge cases when pulse events occur at the same time.
  ({user}`grahamgower`, {issue}`328`, {issue}`357`, {pr}`362`)
- `todict_simplified()` now properly handles symmetric migrations when the
  end times of the demes involved differ.
  ({user}`apragsdale`, {issue}`384`, {pr}`385`)

## 0.1.2 - 2021-06-08

**New features**:

- Add `Graph.migration_matrices()` to get the migration matrices for a graph.
  ({user}`grahamgower`, {issue}`309`, {pr}`320`)
- Add `Deme.size_at()` to get the size of a deme at a given time.
  ({user}`grahamgower`, {issue}`312`, {pr}`314`)
- Support "linear" as an `Epoch.size_function`.
  ({user}`noscode`, {issue}`296`, {pr}`310`)
- Downstream test code can now use the `demes.hypothesis_strategies.graphs()`
  [hypothesis](https://hypothesis.readthedocs.io/) strategy to generate a
  random `Graph`. This is preliminary, and as such is not yet documented,
  but is used for testing internally with some success. The API may change
  in the future in response to requests from downstream application authors.
  ({user}`grahamgower`, {issue}`217`, {pr}`294`)
- The string representation for a graph, `Graph.__str__()`, is now the
  simplified YAML output.
  ({user}`grahamgower`, {issue}`235`, {pr}`293`)

**Breaking changes**:

- The undocumented msprime and stdpopsim converters have been removed.
  ({user}`grahamgower`, {issue}`313`, {pr}`316`)
- The JSON spec doesn't allow serialising infinite float values (although the
  Python json library does support this by default). So for JSON output we
  instead use the string "Infinity".
  ({user}`grahamgower`,
  [demes-spec#70](https://github.com/popsim-consortium/demes-spec/issues/70),
  {pr}`311`)

## 0.1.1 - 2021-04-21

Remove the "demes" console_scripts entry point.
This isn't documented/supported and was left in accidentally.


## 0.1.0 - 2021-04-19

**Breaking changes**:

- The interpretation has been changed for symmetric migrations when the
  `start_time` (and/or `end_time`) is not specified. Symmetric migrations are
  now resolved separately for each pair in the list of participating demes.
  To accommodate this semantic change, the `SymmetricMigration` class has
  been removed, and symmetric migrations are always resolved into pairs of
  `AsymmetricMigration` objects.
  ({user}`grahamgower`, {issue}`263`, {pr}`268`)

- The `size_function` field can no longer be an arbitrary string.
  Only the "constant" and "exponential" strings are recognised.
  ({issue}`262`, {pr}`278`)

**New features**:
- The `from_ms()` function has been added to convert an ms command line
  into a `Graph`.
  ({user}`jeromekelleher`, {user}`grahamgower`, {issue}`74`, {pr}`102`)

**Bug fixes**:
- `Graph.in_generations()` no longer changes time values for a graph
  when `time_units == "generations"` and `generation_time is not None`.
  ({user}`grahamgower`, {issue}`273`, {pr}`274`)

## 0.1.0a4 - 2021-03-22

**Breaking changes**:

- The deme `id` field has been renamed to `name`. This applies to both
  the data model (YAML files) and the `Deme` class.
  ({pr}`246`, discussion at https://github.com/popsim-consortium/demes-spec/issues/59)

**Bug fixes**:

- Check for multiple pulses causing ancestry proportion > 1.
  ({user}`grahamgower`, {issue}`250`, {pr}`251`)
- Check selfing_rate + cloning_rate <= 1.
  ({user}`grahamgower`, {issue}`242`, {pr}`251`)
- Check for pulse time edge cases.
  ({user}`grahamgower`, {issue}`243`, {pr}`249`)
- Check sum of migration rates entering a deme are <= 1.
  ({user}`grahamgower`, {issue}`244`, {pr}`249`)
- Fix migration.end_time in convert.from_msprime.
  ({user}`grahamgower`, {pr}`241`)


## 0.1.0a3 - 2021-02-25

**Bug fixes**:

- Fix `Graph.in_generations()` to also convert the `Deme.start_time` field.
  Thanks to {user}`apragsdale` for reporting the problem.
  ({user}`grahamgower`, {issue}`224`, {pr}`225`).
- Fix `assert_close()` and `is_close()` equality checks to compare the deme
  `start_time`.
  ({user}`grahamgower`, {issue}`224`, {pr}`225`).


## 0.1.0a2 - 2021-02-24

Alpha release for testing. The API and the schema for YAML files have been
largely agreed upon. Backwards-incompatible changes before the first stable
release are still possible, but are considered unlikely.


## 0.1.0a1 - 2020-11-12

Initial alpha release to reserve the name 'demes' on pypi.
