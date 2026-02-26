from django.contrib import admin
from .models import Experiment

class ExperimentAdmin(admin.ModelAdmin):
    fields = ('name', 'description', 'other_fields_here')
    help_texts = {
        'name': 'The experiment name must match the directory name in runner/experiments.',
    }

admin.site.register(Experiment, ExperimentAdmin)