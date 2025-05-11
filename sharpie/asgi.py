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

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sharpie.settings")
# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

from example import websocket as example_websocket
from mountain import websocket as mountain_websocket
from AMaze import websocket as amaze_websocket
from spread import websocket as spread_websocket
from tag import websocket as tag_websocket
#from minecraft import websocket as minecraft_websocket

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(
                URLRouter([
                        re_path("example/run", example_websocket.Consumer.as_asgi()),
                        re_path("mountain/run", mountain_websocket.Consumer.as_asgi()),
                        re_path("AMaze/run", amaze_websocket.Consumer.as_asgi()),
                        re_path("spread/run", spread_websocket.Consumer.as_asgi()),
                        re_path("tag/run", tag_websocket.Consumer.as_asgi()),
                        #re_path("minecraft/run", minecraft_websocket.Consumer.as_asgi()),
                    ]
                )
            )
        ),
    }
)