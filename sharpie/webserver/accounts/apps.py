"""Django app configuration for the accounts app."""
from django.apps import AppConfig

class AccountsConfig(AppConfig):
    """App configuration for the accounts app."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sharpie.webserver.accounts'  # Full path here