from django.contrib import admin
from .models import Experiment, Trial, Interaction

class ExperimentAdmin(admin.ModelAdmin):
    list_display = ['name', 'description', 'input_list', 'user_number']

class TrialAdmin(admin.ModelAdmin):
    list_display = ['experiment', 'room_name', 'user', 'agent_played', 'started_at', 'ended_at']

class InteractionAdmin(admin.ModelAdmin):
    list_display = ['trial', 'timestamp', 'step', 'observations', 'actions', 'rewards']

admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Trial, TrialAdmin)
admin.site.register(Interaction, InteractionAdmin)