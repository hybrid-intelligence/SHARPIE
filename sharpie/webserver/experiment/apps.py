"""Django app configuration for the experiment app."""
from django.apps import AppConfig

class AccountsConfig(AppConfig):
    """App configuration for the experiment app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sharpie.webserver.experiment'  # Full path here