Usage
=====

.. _installation:

Installation
------------

To use SHARPIE, we highly recommend to use a virtual environment such as Anaconda. If you have already installed Anaconda:

.. code-block:: console

   conda create -n sharpie_env python=3.11
   conda activate sharpie_env

Then, git clone the SHARPIE repository:

.. code-block:: console

   git clone https://github.com/hybrid-intelligence/SHARPIE.git

Install Redis server:

.. code-block:: console

   # On Ubuntu
   sudo apt-get install redis-server & redis-server

Navigate to the SHARPIE directory and install the required packages:

.. code-block:: console

   cd SHARPIE
   pip install -r requirements.txt   # This includes django-extensions and pygraphviz

   # If pygraphviz fails to install, install system dependencies first:
   sudo apt-get install graphviz libgraphviz-dev
   pip install -r requirements.txt

Create a database file (SQLite by default) and add an admin user:

.. code-block:: console

   cd webserver
   python manage.py makemigrations accounts experiment data runner
   python manage.py migrate
   python manage.py createsuperuser

Generating the data model diagram
-----------------------------------

To regenerate the data model documentation diagram after making changes to the Django models:

.. code-block:: console

   cd webserver
   python manage.py graph_models accounts experiment data runner -o ../docs/source/_static/data_model.png

This requires `django-extensions` and `pygraphviz` to be installed, which are included in the project's requirements.

Run in development mode
----------------

Start the web server:

.. code-block:: console

   cd webserver
   python manage.py runserver

Go to the admin interface and add a new runner with the desired connection key (e.g., "secret"). In another terminal, start the runner:

.. code-block:: console

   cd runner
   python manage.py runserver --connection-key=secret

You can access the website at http://localhost:8000 and manage the authorized users from http://localhost:8000/admin with the username and password that you set at the end of the installation. For now there is no experiment available but you can find some examples ready to use in our `galery <https://github.com/hybrid-intelligence/SHARPIE_Gallery/>`_!

Run in production mode
------------------

For the web server:
Start by looking at the `deployment checklist <https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/>`_ from Django. We recommend using the `example setup <https://channels.readthedocs.io/en/latest/deploying.html#example-setups>`_ with Nginx and Supervisor from the Channels documentation.

You can find an example supervisor configuration file in `deployment/webserver_supervisor.conf`. This configuration creates the `/run/daphne` directory before starting Daphne to prevent socket failures on systems that periodically clean `/run`. Modify the paths to match your configuration and copy it to `/etc/supervisor/conf.d/`. Then, run::

   sudo supervisorctl reread
   sudo supervisorctl update

For the runner:
We recommend using `supervisor <http://supervisord.org/>`_ to manage the runner process. You can find an example configuration file in `deployment/runner_supervisor.conf`. You can modify the paths mentioned in the file to match your configuration and copy it to `/etc/supervisor/conf.d/`. Then, run::

   sudo supervisorctl reread
   sudo supervisorctl update

Updating your installation
------------------
If you already have a release of SHARPIE installed, you can upgrade it by downloading the latest version from GitHub, copy your settings (and database file if you are using SQLite) to your new installation directory, and run:

.. code-block:: console

   cd webserver
   python manage.py makemigrations accounts experiment data runner
   python manage.py migrate

This will look at the migrations files under /accounts and /experiment, and apply any new migrations that are available to your database.