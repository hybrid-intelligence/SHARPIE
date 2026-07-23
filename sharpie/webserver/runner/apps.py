"""Django app configuration for the runner app."""
from django.apps import AppConfig

class AccountsConfig(AppConfig):
    """App configuration for the runner app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sharpie.webserver.runner'  # Full path here