from django.urls import path
from . import views

# These are the default URLs available:
# - 2 URLs redirected to 1 view for the configuration
# - 1 URL redirected to 1 view for the actual experiment
urlpatterns = [
    path("<str:link>/config", views.config_, name="config"),
    path("<str:link>/run/<str:room>", views.run_, name="run"),
    path("download/policy_template", views.download_policy_template, name="download_policy_template"),
    path("download/environment_template", views.download_environment_template, name="download_environment_template"),
]