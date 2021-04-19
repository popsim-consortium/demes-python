# Changelog

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
