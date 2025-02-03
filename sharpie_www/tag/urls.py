from django.urls import path

from . import views

urlpatterns = [
    path("", views.config_, name="index"),
    path("config", views.config_, name="config"),
    path("run", views.run_, name="run"),
]