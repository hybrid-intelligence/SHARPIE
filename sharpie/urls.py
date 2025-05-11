"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("", include("home.urls")),
    path("example/", include("example.urls")),
    path("mountain/", include("mountain.urls")),
    path("AMaze/", include("AMaze.urls")),
    path("spread/", include("spread.urls")),
    path("tag/", include("tag.urls")),
    #path("minecraft/", include("minecraft.urls")),
    path('accounts/', include("accounts.urls")),
    path('admin/', admin.site.urls),
]
