from django.contrib import admin
from .models import Experiment, Policy, Agent, Environment

class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('name', 'enabled')
    search_fields = ('name',)
    list_filter = ('enabled',)

class PolicyAdmin(admin.ModelAdmin):
    list_display = ('name', 'filepaths')
    search_fields = ('name',)

class AgentAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'policy', 'participant')
    search_fields = ('name',)
    fieldsets = (
        (None, {'fields': ('role', 'name', 'description', 'policy', 'participant')}),
        ('Inputs', {'fields': ('keyboard_inputs', 'keyboard_input_display', 'multiple_keyboard_inputs', 'inputs_type', 'textual_inputs')}),
        (None, {'fields': ('metadata',)}),
    )

class EnvironmentAdmin(admin.ModelAdmin):
    list_display = ('name', 'filepaths')
    search_fields = ('name',)

admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Policy, PolicyAdmin)
admin.site.register(Agent, AgentAdmin)
admin.site.register(Environment, EnvironmentAdmin)
