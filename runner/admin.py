from django.contrib import admin
from .models import Experiment

class ExperimentAdmin(admin.ModelAdmin):
    fields = ['name', 'description']
    help_texts = {
        'name': 'The entry name must match the experiment directory in runner/experiments.',
    }

admin.site.register(Experiment, ExperimentAdmin)