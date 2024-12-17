from django.urls import path

from . import views

urlpatterns = [
    path("", views.index_, name="index"),
    path("continue", views.continue_, name="continue"),
    path("restart", views.restart_, name="restart"),
]