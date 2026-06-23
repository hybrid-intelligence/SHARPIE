Updating SHARPIE
================
| If you already have a release of SHARPIE installed, you can upgrade it.
| We advize to first make a backup of your SHARPIE installation, including of your data.


Now update to the newest version of SHARPIE using pip, and update your database:

.. code-block:: console

   pip install sharpie --upgrade
   sharpie-web makemigrations
   sharpie-web migrate

This will look at the data migrations files, and apply any new migrations that are available to your database.
