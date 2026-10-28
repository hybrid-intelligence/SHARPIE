"""Admin site configuration for experiments, policies, agents, and environments."""
from django.contrib import admin, messages
from django.http import HttpResponse
from django.utils.text import get_valid_filename

from sharpie.install.exporter import ExportError, export_experiment_yaml

from .models import Experiment, Policy, Agent, Environment, ConnectionCheckerConfig


@admin.action(description="Export selected experiment as YAML", permissions=["view"])
def export_as_yaml(modeladmin, request, queryset):
    """Download the selected experiment as a config.yaml that sharpie-install accepts."""
    # A config.yaml describes a single experiment
    if queryset.count() != 1:
        modeladmin.message_user(request, "Select exactly one experiment to export.",
                                level=messages.ERROR)
        return None

    experiment = queryset.get()
    try:
        content = export_experiment_yaml(experiment)
    except ExportError as e:
        modeladmin.message_user(request, str(e), level=messages.ERROR)
        return None

    filename = get_valid_filename(f"{experiment.link}.yaml")
    response = HttpResponse(content, content_type='application/yaml; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


class ExperimentAdmin(admin.ModelAdmin):
    """Admin configuration for Experiment: list, search, and filter experiments by name and enabled status."""
    list_display = ('name', 'enabled')
    search_fields = ('name',)
    list_filter = ('enabled',)
    actions = [export_as_yaml]

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
