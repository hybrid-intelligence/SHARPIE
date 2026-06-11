#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
# import webserver.mysite

def main():
    """Run administrative tasks."""
    # Note FdH: this breaks running manage.py from `sharpie/webserver` but enables running it as a pip installable package
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sharpie.webserver.server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
