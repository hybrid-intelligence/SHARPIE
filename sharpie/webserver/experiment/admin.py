"""Admin site configuration for experiments, policies, agents, and environments."""
from django.contrib import admin
from .models import Experiment, Policy, Agent, Environment, ConnectionCheckerConfig

class ExperimentAdmin(admin.ModelAdmin):
    """Admin configuration for Experiment: list, search, and filter experiments by name and enabled status."""
    list_display = ('name', 'enabled')
    search_fields = ('name',)
    list_filter = ('enabled',)

class PolicyAdmin(admin.ModelAdmin):
    """Admin configuration for Policy: list and search policies by name."""
    list_display = ('name', 'filepaths')
    search_fields = ('name',)

class AgentAdmin(admin.ModelAdmin):
    """Admin configuration for Agent: list and search agents, with fields grouped into general info and inputs."""
    list_display = ('name', 'description', 'policy', 'participant')
    search_fields = ('name',)
    fieldsets = (
        (None, {'fields': ('role', 'name', 'description', 'policy', 'participant')}),
        ('Inputs', {'fields': ('keyboard_inputs', 'keyboard_input_display', 'multiple_keyboard_inputs', 'inputs_type', 'textual_inputs')}),
        (None, {'fields': ('metadata',)}),
    )

class EnvironmentAdmin(admin.ModelAdmin):
    """Admin configuration for Environment: list and search environments by name."""
    list_display = ('name', 'filepaths')
    search_fields = ('name',)

class ConnectionCheckerConfigAdmin(admin.ModelAdmin):
    """Admin configuration for ConnectionCheckerConfig: enforces the singleton constraint in the admin UI."""
    list_display = ('bandwidth_threshold', 'latency_threshold', 'test_image_size')

    def has_add_permission(self, request):
        """Allow adding a new config only if no instance exists yet."""
        return not self.model.objects.exists()

    def has_delete_permission(self, request, obj=None):
        """Never allow deleting the config instance from the admin."""
        return False

admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Policy, PolicyAdmin)
admin.site.register(Agent, AgentAdmin)
admin.site.register(Environment, EnvironmentAdmin)
admin.site.register(ConnectionCheckerConfig, ConnectionCheckerConfigAdmin)
