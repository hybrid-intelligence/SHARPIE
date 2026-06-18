Usage
=====

.. _installation:

Installation
------------

First install the required system dependencies. For Ubuntu, you can run

.. code-block:: console

   # On Ubuntu
   sudo apt-get install redis-server && sudo systemctl enable redis-server && redis-server

Verify that redis has installed and runs correctly: 

.. code-block:: console

      redis-server --version # should output >8
      redis-cli ping # should output "PONG"

To use SHARPIE, we highly recommend to use a virtual environment such as Anaconda. If you have already installed Anaconda:

.. code-block:: console

   conda create -n sharpie_env python=3.11
   conda activate sharpie_env

Install SHARPIE using pip. You can either install the latest release from PyPI:

.. code-block::console

   pip install sharpie

Or install the latest development version from the GitHub repository:
.. code-block:: console

   git clone https://github.com/hybrid-intelligence/SHARPIE.git
   cd SHARPIE
   pip install .  # if you want to use SHARPIE
   pip install -e .  # if you want to edit SHARPIE
   pip install -e .[dev] # if you want to edit SHARPIE and contribute to the codebase

   # If pygraphviz fails to install, install system dependencies first:
   sudo apt-get install graphviz libgraphviz-dev

Create a database file (SQLite by default) and add an admin user:

.. code-block:: console

   sharpie-web migrate
   sharpie-web createsuperuser

Run in development mode
----------------

Start the web server:

.. code-block:: console

   sharpie-web runserver

Go to the admin interface at ``localhost:8000/admin/`` and log in with your superuser name and password.

Add a new runner ``Runner > Runners > add`` and choose a connection key (e.g., "my_secret"). Select "SAVE".

Create a new consent form ``Accounts > Contents > add``. Fill in the required forms and select "SAVE".

Open a new terminal in the SHARPIE root directory, and start the runner:

.. code-block:: console

   conda activate sharpie_env
   sharpie-runner runserver --connection-key=my_secret

The terminal running the webserver should now show log a websocket connection between the runner and the webserver.

You can access the website at http://localhost:8000 and manage the authorized users from http://localhost:8000/admin with the username and password that you set at the end of the installation. For now there is no experiment available but you can find some examples ready to use in our `gallery <https://github.com/hybrid-intelligence/SHARPIE_Gallery/>`_!

Hello world example
------------------
In this minimal example, we will create a random mountain car environment with human controls.

In a clean directory create a new file called `environment.py` and copy the following code:

.. code-block:: python

   import gymnasium as gym

   class EnvironmentWrapper:
      """Wrapper for the MountainCar-v0 Gymnasium environment."""

      def __init__(self):
         """Initialize the environment."""
         self.env = gym.make("MountainCar-v0", render_mode="rgb_array")

      def reset(self):
         """
         Reset the environment to an initial state.

         Returns:
               observation: Initial observation (numpy array)
               info: Additional information
         """
         observation, info = self.env.reset()
         return observation, info

      def step(self, action_dict):
         """
         Execute one step in the environment.

         Args:
               action_dict: Dictionary with agent id as keys and action as value

         Returns:
               observation: New observation (numpy array)
               reward: Reward for the action (float)
               terminated: Whether the episode has ended (bool)
               truncated: Whether the episode was truncated (bool)
               info: Additional information (dict)
         """
         # Convert dict action to discrete action (int) - MountainCar is single-agent
         action = list(action_dict.values())[0]
         observation, reward, terminated, truncated, info = self.env.step(action)
         return observation, reward, terminated, truncated, info

      def render(self):
         """
         Render the environment.

         Returns:
               image: Rendered image of the environment (numpy array)
         """
         image = self.env.render()
         return image


   environment = EnvironmentWrapper()

Now add your random agent:

.. code-block:: python
   import gymnasium as gym

   class Policy:
      """Random policy for the MountainCar-v0 environment."""

      def __init__(self):
         """Initialize the policy."""
         self.action_space = gym.make("MountainCar-v0").action_space

      def predict(self, observation, **kwargs):
         """
         Select an action based on the observation.

         Args:
               observation: Current observation (numpy array)

         Returns:
               action: Selected action (int)
         """
         return self.action_space.sample()
   
   policy = Policy()

Now browse to http://localhost:8000/admin and create the following entries:

Experiment > Environments > Add Environment > Name: "MountainCar", paste the following list of environment files:
Get `/full/path/to` by running `pwd` in the terminal

.. code-block:: json
   
   {
      "enviroment": "/full/path/to/environment.py",
   }

Experiment > Policies > Add Policy > Name: "Random Policy", paste the following in list of policy files:

.. code-block:: json

   {
      "policy": "/full/path/to/policy.py",
   }

Experiment > Agents > Add Agent > Role: "random_agent", Name: "Random Agent" Policy: "Random Policy", "Can the participant act?": yes.

Leave all other fields as default and select "SAVE". Then, create a new experiment:

Experiment > Experiments > Add Experiment > Name: "Hello world Experiment", Link: "hello-world", Home page tet: "home test", Experiment page text: "experiment test".
Link it to the environment and agent you just created, set the number of episodes to 30 and select "SAVE".

If you have already started your runner, stop it and install gymnasium[classic_control] and restart it with your connection key:


.. code-block:: console
   
   pip install "gymnasium[classic_control]"
   sharpie-runner runserver --connection-key my_secret


Now browse to http://localhost:8000 and click on your experiment and start it.
You should see the MountainCar environment rendered and be able to control the car with the left, down, and right arrow keys!

Run in production mode
------------------

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

Updating your installation
------------------
If you already have a release of SHARPIE installed, you can upgrade it by downloading the latest version from GitHub, copy your settings (and database file if you are using SQLite) to your new installation directory, and run:

.. code-block:: console

   cd webserver
   sharpie-web makemigrations accounts experiment data runner
   sharpie-web migrate

This will look at the migrations files under /accounts and /experiment, and apply any new migrations that are available to your database.

Generating the data model diagram
-----------------------------------

To regenerate the data model documentation diagram after making changes to the Django models:

.. code-block:: console

   cd webserver
   sharpie-web graph_models accounts experiment data runner -o ../docs/source/_static/data_model.png

This requires `django-extensions` and `pygraphviz` to be installed, which are included in the project's requirements.
