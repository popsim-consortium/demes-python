1. Prepare for the release by updating the `CHANGELOG.md`. The version here
should have the format MAJOR.MINOR.PATCH, or for a beta release
MAJOR.MINOR.PATCHbX, e.g. 1.0.0b1. Commit the changes to the changelog;
make pull request; merge.

2. Tag and push release. E.g. `git tag 1.0.0b1` and `git push origin 1.0.0b1`.
This triggers the `wheel.yaml` github workflow that uploads
the release to https://test.pypi.org/project/demes/ and can be
installed with:
`pip install demes --upgrade --pre --extra-index-url https://test.pypi.org/simple/`.

3. Create a release in the GitHub UI, based on the tag that was pushed
(paste the output of `docs/convert_changelog.py` into the release notes).
This will trigger the `wheel.yaml` workflow again, and upload the release
to pypi.org. If you don't wish to publish a beta release, this step can
be omitted.

4. Within a few hours, the conda-forge infrastructure will automagically detect
the new release on PyPI and will make a pull request against the
[demes-feedstock](https://github.com/conda-forge/demes-feedstock/) repository
to bump the version number. If there have been changes to dependencies,
the conda recipe may need to be fixed by adding commits onto the bot's
pull request. Once satisfied, merge the pull request and the package will
be conda-installable after a few minutes.
See the [conda-forge maintainer docs](
https://conda-forge.org/docs/maintainer/updating_pkgs.html)
for additional information.
