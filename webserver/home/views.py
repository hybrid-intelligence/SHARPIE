from django.shortcuts import render
from experiment.models import Experiment


def index(request):
    experiments = Experiment.objects.all()
    return render(request, "home/index.html", {"experiments": experiments})