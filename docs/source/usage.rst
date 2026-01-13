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
   pip install -r requirements.txt

Create a database file (SQLite by default) and add an admin user:

.. code-block:: console

   cd webserver
   python manage.py migrate
   python manage.py createsuperuser

Run in development mode
----------------

Start the web server:

.. code-block:: console

   cd webserver
   python manage.py runserver

In another terminal, start the runner:

.. code-block:: console

   cd runner
   python manage.py runserver

You can access the website at http://localhost:8000 and manage the authorized users from http://localhost:8000/admin with the username and password that you set at the end of the installation. For now there is no experiment available but you can find some examples ready to use in our `galery <https://github.com/hybrid-intelligence/SHARPIE_Gallery/>`_!

Run in production mode
------------------

For the web server:
Start by looking at the `deployment checklist <https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/>`_ from Django. We recommenf using the `example setup <https://channels.readthedocs.io/en/latest/deploying.html#example-setups>`_ with Nginx and Supervisor from the Channels documentation.

For the runner:
We recommend using `supervisor <http://supervisord.org/>`_ to manage the runner process. You can find an example configuration file in `runner/runner_supervisor.conf`. You can modify the paths mentioned in the file to match your configuration and copy it to `/etc/supervisor/conf.d/`. Then, run: `sudo supervisorctl reread`_ and `sudo supervisorctl update`_ to apply the changes.