Release Checklist
=================

This non-exhaustive checklist is meant to help with releasing a new version of SHARPIE:

* check whether all tests pass
* check documentation builds
* check gallery installs works
* check that the performance has not regressed using the benchmark
* update the version in ``docs/source/conf.py``
* update the version in ``pyproject.toml``
* create build, upload to pypi with twine
* ensure the new version is available on `readthedocs <https://sharpie.readthedocs.io/>`_

Release Process
---------------

First create a release candidate by creating a new branch ``x-x-xrc1``.

| Test this release candidate locally and obtain some additional testing from at least two collaborators.
| Ensure tests are performed on all supported OSes.

Once the release candidate is bug-free, tag the latest commit of that branch wit ``x-x-x``, create a new build, and test this again locally.
Now upload this build using twine.
