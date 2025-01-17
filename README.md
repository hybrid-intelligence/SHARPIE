# SHARPIE
## Shared Human-AI Reinforcement Learning Platform for Interactive Experiments
Our framework is relying on Django for serving files to the user browser. Therefore, it is following the architecture of Django to organize the code i.e. decomposition of the pages in apps.

In this repository, each use-case described in our paper is a separate app, which translates to a different folder. For clarity we will detail briefly here the important files in our project but we highly recommend to look at the Django documentation for a better understanding.

### Common files for all apps
* `manage.py` is the framework manager, it allows you to easily launch your website and refreshes automatically to perform live testing.
* `mysite` is the folder holding the configuration of your website. `urls.py` declares which apps are available and what is their base url on your website. `settings.py` declares all the Django libraries you are going to use as well as where the HTML templates or the static files (e.g. CSS, JS, images, etc) are located in your project.
* `mysite/template` in the folder holding all the HTML template used on the website. Each app has a different folder which is name exactly as the app's folder. Those templates allow you decide of the look of your pages.
* `mysite/template/base.html` defines the menu as well as the footer of the page and is included in all other templates.

### Specific files for an app
* `urls.py` declares which pages are available in your app and what is their url on your website.
* `views.py` is the core of your app and defines how it will interact with the user. The following fonctions are always present:
  * `config_`, defines the configuration needed for the app. It is coupled with a template that displays and can asks the user to modify it.
  * `train_`, to restart the RL environment and save that we are in a training mode. It is coupled with a template that displays the current state of the environment and catpures the input from the user.
  * `evaluate_`, to restart the RL environment and save that we are in a evaluation mode. It is coupled with a template that displays the current state of the environment and catpures the input from the user.
  * `step_`, to perform a step in the RL environment. This is only a REST endpoint and is not coupled with a template.
  * `log_`, to log any kind of information about the interaction with the user. This will be very usefull for the experimenter in order.
