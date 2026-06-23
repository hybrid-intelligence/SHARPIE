Contributing
============

.. warning:: Under construction

We strive for SHARPIE to have a healthy community that works *with* and works *on* SHARPIE. You can contribute in many ways, either on the framework itself or in the wider ecosystem.

Development Installation
------------------------
SHARPIE can only be installed in development mode using pip:

.. code-block:: console

   git clone https://github.com/hybrid-intelligence/SHARPIE.git
   cd SHARPIE
   pip install -e .[docs] # if you want to contribute to the documentation
   pip install -e .[dev] # if you want to edit SHARPIE and contribute a bugfix
   pip install -e .[dev,docs] # if you want to contribute a new feature or improvement, you will likely need to update the docs

SHARPIE cannot be run directly without installation because of fully specified import paths that are required for packaging and distribution using pip.

Development Server
------------------
| If you are familiar with Django, you may expect to launch the development server by running ``python manage.py runserver`` from the ``sharpie/webserver/`` directory.
| SHARPIE, however, is distributed as a package which requires dedicated paths, which is not compatible with running from the ``sharpie/webserver`` directory.

| SHARPIE therefore provides a dedicated entry point ``sharpie-web runserver`` that works wherever for convenience.
| This command is specified in ``pyproject.toml`` in the ``[project.scripts]`` block.

Documentation
-------------
The documentation source files are located in ``docs/source`` and can be built using `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_.

To build the documentation locally, make sure you have installed sharpie using the ``[docs]`` options.

.. code-block:: console

   cd docs/source
   make clean # removes any old build
   make html  # creates HTML files

Now open ``docs/build/html/index.html`` in your browser to view your local documentation build.

To have them dynamically generated with auto-browser refresh instead use

.. code-block:: console

    make livehtml


Look for the URL in the terminal output, it is likely to be `127.0.0.1:8000 <http://127.0.0.1:8000>`_.

Tests
----------
.. warning:: Under construction

Packaging and Distribution
--------------------------

To build the SHARPIE package, from the project root:

.. code-block:: console

  rm -rf build/ dist/ *.egg-info src/*.egg-info && python -m build

To distribute a new release, ensure you have completed the release checklist, obtain credentials for the sharpie pypi project and use twine to distribute.


Contributors Guide
------------------
.. include:: ../../CONTRIBUTING.md
   :parser: myst_parser.parsers.docutils_
