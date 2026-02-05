from django.contrib import admin
from .models import Experiment

class ExperimentAdmin(admin.ModelAdmin):
    fields = ['name', 'description']
    help_texts = {
        'name': 'This name must match the directory name in runner/experiments for the experiment to run correctly.'
    }

admin.site.register(Experiment, ExperimentAdmin)