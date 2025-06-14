from django.urls import path
from . import views

app_name = 'mountain'

# These are the default URLs available:
# - 1 URL for the task description (index)
# - 1 URL for the configuration
# - 1 URL for the actual experiment
urlpatterns = [
    path("", views.task_description_, name="index"),
    path("config/", views.config_, name="config"),
    path("run/", views.run_, name="run"),
]