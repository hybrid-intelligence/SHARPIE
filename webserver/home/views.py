from django.shortcuts import render
from experiment.models import Experiment

from mysite.settings import DEMO


def index(request):
    experiments = Experiment.objects.all()
    return render(request, "home/index.html", {"experiments": experiments, "DEMO": DEMO})