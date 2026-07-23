"""Django app configuration for the home app."""
from django.apps import AppConfig

class AccountsConfig(AppConfig):
    """App configuration for the home app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sharpie.webserver.home'  # Full path here