from django.urls import path
from . import views

# These are the default URLs available:
# - 2 URLs redirected to 1 view for the configuration
# - 1 URL redirected to 1 view for the actual experiment
urlpatterns = [
    path("", views.config_, name="index"),
    path("config", views.config_, name="config"),
    path("experiment/TAMERmountain/run", views.run_tamer_, name="runtamer"),
    path("run", views.run_, name="run"),
    path("evaluate", views.evaluate_, name="evaluate"),
]