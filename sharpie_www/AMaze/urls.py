from django.urls import path

from . import views

urlpatterns = [
    path("", views.index_, name="index"),
    path("stored_policy", views.stored_policy_, name="Stored policy"),
    path("restart", views.restart_, name="Restart"),
    path("continue", views.continue_, name="Continue"),
    path("evaluate", views.evaluate_, name="Evaluate"),
]
