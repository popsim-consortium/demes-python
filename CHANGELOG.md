# Changelog

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
