from django.contrib import admin
from .models import Info

class InfoAdmin(admin.ModelAdmin):
    list_display = ('user', 'room', 'reward', 'changed')  # specify fields to display in the admin table
    list_filter = ('user', 'room', 'reward', 'changed')  # specify fields to filter in the admin table
    search_fields = ['user']  # enable searching by name field

admin.site.register(Info, InfoAdmin)  # register your model with the custom admin class