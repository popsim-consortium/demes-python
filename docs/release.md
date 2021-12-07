1. Prepare for the release by updating the `CHANGELOG.md`. The version here
should have the format MAJOR.MINOR.PATCH, or for a beta release
MAJOR.MINOR.PATCHbX, e.g. 1.0.0b1. Commit the changes to the changelog;
make pull request; merge.

2. Tag and push release. E.g. `git tag 1.0.0b1` and `git push origin 1.0.0b1`.
This triggers the `wheel.yaml` github workflow that uploads
the release to https://test.pypi.org/project/demes/ and can be
installed with:
`pip install demes --upgrade --pre --extra-index-url https://test.pypi.org/simple/`.

3. Create a release in the GitHub UI, based on the tag that was pushed.
This will trigger the `wheel.yaml` workflow again, and upload the release
to pypi.org. If you don't wish to publish a beta release, this step can
be omitted.
