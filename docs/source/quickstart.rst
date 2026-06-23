Quickstart
==========

From your project root, start the web server:

.. code-block:: console

   sharpie-web runserver

Go to the admin interface at `localhost:8000 <https://localhost:8000>`_
 and log in with your superuser name and password.

| Add a new runner ``Runner > Runners > add`` and choose a connection key (e.g., "my_secret").
| Select "SAVE".

| Create a new consent form ``Accounts > Contents > add``.
| Fill in the required forms and select "SAVE".

Open a new terminal in the SHARPIE root directory, and start the runner:

.. code-block:: console

   conda activate sharpie_env
   sharpie-runner runserver --connection-key=my_secret

| Now open a third terminal and install example experiments from the Gallery.

.. code-block:: console

    git clone https://github.com/hybrid-intelligence/SHARPIE_Gallery.git /var/www/SHARPIE_Gallery
    sharpie-install --all --gallery-dir /var/www/SHARPIE_Gallery

Now browse to `localhost:8000 <http://localhost:8000>`_ and test your experiment.