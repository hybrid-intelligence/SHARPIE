from django.urls import path
from . import views

app_name = 'tamer_maze'

# These are the default URLs available:
# - 2 URLs redirected to 1 view for the configuration
# - 1 URL redirected to 1 view for the actual experiment
urlpatterns = [
    path("", views.task_description_, name="index"),
    path("config/", views.config_, name="config"),
    path("run/", views.run_, name="run"),
]