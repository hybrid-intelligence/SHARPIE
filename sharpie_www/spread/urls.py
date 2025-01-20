from django.urls import path

from . import views

urlpatterns = [
    path("", views.config_, name="index"),
    path("config", views.config_, name="config"),
    path("train", views.train_, name="train"),
    path("evaluate", views.evaluate_, name="evaluate"),
    path("step", views.step_, name="step"),
    path("log", views.log_, name="log"),
]