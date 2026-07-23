"""Django app configuration for the server app."""
from django.apps import AppConfig

class AccountsConfig(AppConfig):
    """App configuration for the server app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sharpie.webserver.server'  # Full path here