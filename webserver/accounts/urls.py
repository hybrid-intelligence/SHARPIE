from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_, name="index"),
    path("login/", views.login_, name="login"),
    path("register/", views.register_, name="register"),
    path("logout/", views.logout_, name="logout"),
]