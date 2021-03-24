See tskit docs on making a release:
https://tskit.readthedocs.io/en/latest/development.html#releasing-a-new-version

Prepare for the release by updating the `CHANGELOG.md`. The version here
should have the format MAJOR.MINOR.PATCH, or for a beta release
MAJOR.MINOR.PATCHbX, e.g. 1.0.0b1. Commit the changes to the changelog;
make pull request; merge.

Tag release.
 - E.g. `git tag 0.1.0`

Build distribution files, and test on testpypi.
 - See https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives
 - `python -m pip install --upgrade build twine`
 - `python -m build`
 - `twine check dist/*`
 - `twine upload --repository testpypi dist/*`
 - `pip uninstall demes`
 - Check it installs: `pip install --upgrade --pre --index-url https://test.pypi.org/simple/ demes`.

Upload to pypi.
 - `twine upload --repository pypi dist/*`
 - Check it installs: `pip install demes --upgrade --pre`.
 
Push release.
 - E.g. `git push origin 0.1.0`
 - Create a release in the GitHub UI, based on the tag that was pushed.

Finally, update `CHANGELOG.md` to start a section for the next release cycle.
