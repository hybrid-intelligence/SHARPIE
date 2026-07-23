"""Admin site configuration for consent and participant records."""
from django.contrib import admin
from .models import Consent, Participant


class ConsentAdmin(admin.ModelAdmin):
    """Admin configuration for Consent: list and filter consent descriptions by name and creation date."""
    list_display = ['name', 'created_at']
    list_filter = ['name', 'created_at']
    readonly_fields = ['created_at']

class ParticipantAdmin(admin.ModelAdmin):
    """Admin configuration for Participant: list, filter, and search participants by user info."""
    list_display = ['user', 'external_id', 'agreed_at', 'withdrawn_at']
    list_filter = ['user__username', 'user__email']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['consent', 'agreed_at', 'withdrawn_at']

admin.site.register(Consent, ConsentAdmin)
admin.site.register(Participant, ParticipantAdmin)