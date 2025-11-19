from django.contrib import admin
from .models import Consent


class ConsentAdmin(admin.ModelAdmin):
    list_display = ['user', 'agreed', 'agreed_at', 'ip_address']
    list_filter = ['agreed', 'agreed_at']
    search_fields = ['user__username', 'user__email', 'ip_address']
    readonly_fields = ['agreed_at']


admin.site.register(Consent, ConsentAdmin)

