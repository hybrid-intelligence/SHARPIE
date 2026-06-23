Development
===========

.. warning:: Under construction

In development mode, SHARPIE is better installed by cloning the repo and installing with all dependencies

.. code-block:: console

    git clone https://github.com/hybrid-intelligence/SHARPIE.git
    pip install -e .[dev,docs]

Documentation
^^^^^^^^^^^^^

To build the docs locally you can use either

.. code-block:: console

    make html

from the `docs` folder to generate files in `docs/_build/html`.
To have them dynamically generated with auto-browser refresh instead use

.. code-block:: console

    make livehtml

.. include:: ../../CONTRIBUTING.md
   :parser: myst_parser.parsers.docutils_
