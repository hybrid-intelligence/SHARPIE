"""Django app configuration for the data app."""
from django.apps import AppConfig

class AccountsConfig(AppConfig):
    """App configuration for the data app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sharpie.webserver.data'  # Full path here