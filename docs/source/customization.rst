Customization
=============

When the different configuration options are insufficient for your needs, you can customize the sharpie code.

We advise to first create a fork of the latest stable release of SHARPIE, see SHARPIE tags on GitHub.

Next, follow the regular installation steps and make sure to install in editable mode using ``pip install -e .`` in the SHARPIE directory.

You can run the sharpie webserver and runner as usual, any changes you make to the SHARPIE source files you installed will be reflected when using the standard SHARPIE commands.

UI elements and Templates
-------------------------
Most of the UI elements and templates are located in one of the apps (directories) in the  ``sharpie/webserver`` directory:

| UI elements and templates are located in ``sharpie/webserver/server`` and ``sharpie/webserver/experiment``, with the exception of the home page which is in ``home/``.
| To add custom interaction elements such as buttons, inputs, images, etc., change the files in the ``templates/`` and ``static/`` directories for these apps.

+++++++++
Templates
+++++++++
We here give an overview of the role of different templates that you may want to choose for customization, all located in ``sharpie/webserver/``:

* ``server/templates/base.html`` the root template, containing the root ``<html>`` tags and the menu of all pages.
* ``home/templates/home/index.html`` the landing page that by default gives an overview of all installed experiments.
* ``experiment/templates/experiment/config.html`` participant-facing experiment configuration page, with room name and agent role selection
* ``experiment/templates/experiment/run.html`` template for experiment, including interaction instructions, history etc. Here you can add custom input elements, additional interaction information, custom history etc.

+++++++
Styling
+++++++
| SHARPIE uses the well-established `Bootstrap CSS <https://getbootstrap.com/>`_ framework.
| Some custom styling is additionally defined in ``sharpie/webserver/server/templates/base.html``, and in ``sharpie/webserver/experiment/static/experiment/css/control.css``.

Interaction
-----------
In case you want to change the interaction, you will need to add the required UI elements to the template first and ensure the styling is as you want it.

Additionally, you will want to implement some logic for capturing and processing the inputs for proper interactions.
The javascript files for this are located in

* ``sharpie/webserver/server/experiment/static/experiment/js/mobileControls.js``
* ``sharpie/webserver/server/experiment/static/experiment/js/inputListener.js``

If you want to send different kind of information between the front end, i.e. participant-facing web UI, and the back-end, i.e. the webserver and the runner, read on.

Communication
-------------
To change which/how information is sent between the web UI, the webserver and the runner, have a look at the following files in ``sharpie/webserver/server/experiment``:

* ``static/experiment/js/websocket.js`` implements the front-end experiment websocket logic.
  In particular have a look at ``websocket.onmessage`` which handles how messages from the web server are processed in the front-end.
* ``websocket.py`` implements back-end experiment websocket logic.
  In particular, have a look at ``_handle_action`` for the processing of participant input and ``_handle_broadcast`` for handling of a a new ``step()`` from the runner.

Finally, have a look at ``sharpie/runner/manage.py`` for logic on the runner-side.



