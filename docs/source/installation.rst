Installation
============

Dependencies
------------

First install the required system dependencies. For Ubuntu:

.. code-block:: console

   # On Ubuntu
   sudo apt-get install redis-server && sudo systemctl enable redis-server && redis-server

Verify that redis has installed and runs correctly: 

.. code-block:: console

      redis-server --version # should output >8
      redis-cli ping # should output "PONG"

Virtual environment
-------------------
Set up a virtual environment such as anaconda.
If you have already installed Anaconda:

.. code-block:: console

   conda create -n sharpie_env python=3.11
   conda activate sharpie_env

Installing SHARPIE
------------------

Install SHARPIE using pip. You can either install the latest release from PyPI:

.. code-block:: console

   pip install sharpie

Or install the latest development version from the GitHub repository:

.. code-block:: console

   git clone https://github.com/hybrid-intelligence/SHARPIE.git
   cd SHARPIE
   pip install .  # if you want to use SHARPIE
   pip install -e .  # if you want to edit SHARPIE
   pip install -e .[dev] # if you want to edit SHARPIE and contribute to the codebase

If pygraphviz fails to install, install system dependencies first:

.. code-block:: console

   sudo apt-get install graphviz libgraphviz-dev


Database initialisation
-----------------------

Go to your project root.
If you do not have a project yet, create a new empty directory.

Create a database and add an admin user:

.. code-block:: console

   sharpie-web migrate
   sharpie-web createsuperuser

See the deep dive for more detailes on alternative database configuration and setup for real-world deployments.