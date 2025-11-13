from django.urls import path
from . import views

from mysite.settings import REGISTRATION_KEY

urlpatterns = [
    path("", views.login_, name="index"),
    path("login/", views.login_, name="login"),
    path("logout/", views.logout_, name="logout"),
]

# Only add the registration URL if a registration key is set to a string or is True
if REGISTRATION_KEY:
    urlpatterns.append(path("register/", views.register_, name="register"))