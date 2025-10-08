from django.contrib import admin
from .models import Experiment, Trial, Interaction, Runner, Queue

class ExperimentAdmin(admin.ModelAdmin):
    list_display = ['name', 'type', 'input_list', 'agent_list', 'link']

class TrialAdmin(admin.ModelAdmin):
    list_display = ['experiment', 'room_name', 'user', 'agent_played', 'started_at', 'ended_at']

class InteractionAdmin(admin.ModelAdmin):
    list_display = ['trial', 'timestamp', 'step', 'observations', 'actions', 'rewards']

class RunnerAdmin(admin.ModelAdmin):
    list_display = ['__str__', 'created_at', 'last_active', 'status', 'current_room', 'ip_address']

class QueueAdmin(admin.ModelAdmin):
    list_display = ['__str__', 'experiment', 'room_name', 'users_waiting', 'created_at', 'status']

admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Trial, TrialAdmin)
admin.site.register(Interaction, InteractionAdmin)
admin.site.register(Runner, RunnerAdmin)
admin.site.register(Queue, QueueAdmin)