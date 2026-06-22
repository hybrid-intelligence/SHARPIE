from django.contrib import admin
from .models import Runner

class RunnerAdmin(admin.ModelAdmin):
    list_display = ('connection_key', 'status', 'last_active', 'ip_address')
    list_filter = ('status', 'session__experiment__name')

admin.site.register(Runner, RunnerAdmin)
