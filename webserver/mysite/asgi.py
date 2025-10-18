"""
ASGI config for websocket project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from channels.sessions import SessionMiddlewareStack
from django.core.asgi import get_asgi_application
from django.urls import re_path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

from experiment import websocket

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(
                URLRouter([
                        re_path(r"^experiment/(?P<link>\w+)/evaluate$", websocket.EvaluateConsumer.as_asgi()),
                        re_path(r"^experiment/(?P<link>\w+)/run$", websocket.RunConsumer.as_asgi()),
                        re_path(r"^experiment/queue$", websocket.QueueConsumer.as_asgi()),
                    ]
                )
            )
        ),
    }
)