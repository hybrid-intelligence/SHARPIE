Deployment
==========

This is a brief guide to deploy SHARPIE in a production setting.

For the web server:
Start by looking at the `deployment checklist <https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/>`_ from Django. We recommend using the `example setup <https://channels.readthedocs.io/en/latest/deploying.html#example-setups>`_ with Nginx and Supervisor from the Channels documentation.

You can find an example supervisor configuration file in `deployment/webserver_supervisor.conf`. This configuration creates the `/run/daphne` directory before starting Daphne to prevent socket failures on systems that periodically clean `/run`. Modify the paths to match your configuration and copy it to `/etc/supervisor/conf.d/`. Then, run::

   sudo supervisorctl reread
   sudo supervisorctl update

For Nginx configuration:
You can find an example configuration file in `deployment/nginx.conf`. This configuration includes WebSocket proxy support and SSL setup. Copy it to `/etc/nginx/sites-available/`, create a symlink to `/etc/nginx/sites-enabled/`, and reload Nginx::

   sudo ln -s /etc/nginx/sites-available/nginx.conf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx

For the runner:
We recommend using `supervisor <http://supervisord.org/>`_ to manage the runner process. You can find an example configuration file in `deployment/runner_supervisor.conf`.

**Important:** The runner requires a connection key to authenticate with the webserver. First, create a Runner in Django Admin:

1. Go to http://localhost:8000/admin
2. Navigate to **Runners** and click **Add Runner**
3. Generate a secure key: ``python -c "from secrets import token_urlsafe; print(token_urlsafe(35))"``
4. Enter the connection key and save

Then, update the supervisor config with your connection key by replacing ``YOUR_CONNECTION_KEY`` in the ``command`` line:

.. code-block:: console

   command=sharpie-runner runserver --connection-key=YOUR_ACTUAL_KEY_HERE

Copy the config to supervisor and enable it:

.. code-block:: console

   sudo cp deployment/runner_supervisor.conf /etc/supervisor/conf.d/sharpie-runner.conf
   sudo supervisorctl reread
   sudo supervisorctl update