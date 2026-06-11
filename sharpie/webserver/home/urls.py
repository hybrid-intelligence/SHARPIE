from django.urls import path
from sharpie.webserver.home import views

urlpatterns = [
    path("", views.index, name="index"),
]