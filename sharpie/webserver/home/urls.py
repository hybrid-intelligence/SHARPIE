from django.urls import path

from ....webserver.home import views

urlpatterns = [
    path("", views.index, name="index"),
]