from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("Qlearning", views.q_learning, name="Qlearning"),
]