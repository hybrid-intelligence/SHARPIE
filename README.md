![versions](https://img.shields.io/pypi/pyversions/pybadges.svg) ![motivation-paper](https://img/shields.io/badge/motivating_paper-blue)

# SHARPIE
## Shared Human-AI Reinforcement Learning Platform for Interactive Experiments
Our framework is relying on Django for serving files to the user browser. Therefore, it is following the architecture of Django to organize the code i.e. decomposition of the pages in apps.

In this repository, each use-case described in our paper is a separate app, which translates to a different folder. For clarity we will detail briefly here the important files in our project but we highly recommend to look at the Django documentation for a better understanding.

### Common files for all apps
* `manage.py` is the framework manager, it allows you to easily launch your website and refreshes automatically to perform live testing.
* `mysite/` is the folder holding the configuration of your website.
  * `settings.py` declares all the Django libraries you are going to use, which applications are available, where the HTML templates or the static files (e.g. CSS, JS, images, etc) are located in your project, how those files are served, etc.
  * `urls.py` declares which apps are available and what is their base url on your website
  * `asgi.py` declares how the protocols are handled by the (asynchronous) server and in particular the websockets
  * `websocket.py` declares the WebsocketTemplate that can be used by each app and takes care of syncronization
  * `templates/base.html` defines the menu as well as the footer of the pages and is included in all other templates.
* `db.sqlite3` DB file used by the website.

### Specific files for an app
* `templates/[app_folder]/` holds the HTML templates that are served by your app.
  * `config.html` configuration page of the app.
  * `run.html` experiment page of the app.
* `static/[app_folder]/` holds the static files that are served by your app.
  * `js/script.js` JS code for the interactions on `run.html` and the connection to the websocket of the server.
  * `run/` various files needed during the experiment like the rendered environment or the saved agent policy.
* `urls.py` declares which pages are available in your app. By default, only declares two views (i.e. pages): `config` and `run`.
* `views.py` defines what happens when a user queries a given URL. By default, only defines two views (i.e. pages): `config` and `run`.
  * `config` takes care of displaying, checking and saving in the user's session all the configuration variables needed for the experiment.
  * `run` loads the needed informations for displaying the experiment page.
* `forms.py` defines the forms needed for our experiment. By default only defines `ConfigForm` that is loaded in `views.py/config`
* `websocket.py` main component of your applications that will take care of the communication with the user. By default only defines `Consumer`. 
* `settings.py` settings of your app and at leats app_name and app_folder

## Development installation
* Git clone this repository
* Install the requirements depending on what you want to try
* Uncomment the apps you want to try in `mysite/settings.py` and `mysite/asgi.py`
* Run `python manage.py runserver`

## Create your own app
* Run `sh create_app.sh [app_name] [app_folder]`
* Add the folder name to INSTALLED_APPS in mysite/settings.py
* Add the folder name to the urlpatterns in mysite/urls.py
* Add the folder name to the websocket in mysite/asgi.py
* In your new app, (at least) modify static/js/script.js and websocket.py to suit your needs!

## Run into production
Follow the instructions on Django website and don't forget to change the secret key (otherwise you might get some surprises...).

## Acknowledgements
This research was funded by the [Hybrid Intelligence
Center](https://hybridintelligence-centre.nl), a 10-year programme funded by the Dutch Ministry of
Education, Culture and Science through the Netherlands Organisation for Scientific Research, Grant
No: 024.004.022.

## Citations
When using this project in a scientific publication please cite:
```bibtex
@inproceedings{sharpiecaihu25,
    booktitle = {AAAI Bridge Program Workshop on Collaborative AI and Modeling of Humans},
    title = {{SHARPIE: A Modular Framework for Reinforcement Learning and Human-AI Interaction
    author = {Ayd\in, H{\"{u}}seyin and Godin-Dubois, Kevin and Braz, Libio Goncalvez and den Hengst,
    Floris and Baraka, Kim and {\c{C}}elikok, Mustafa Mert and Sauter, Andreas and Wang, Shihan and
    Oliehoek, Frans A},
    month = {feb},
    address = {Philadelphia, Pennsylvania, USA},
    Experiments}},
    year = {2025}
}
```
