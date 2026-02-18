from django.contrib import admin
from .models import Session, Episode, Record

class SessionAdmin(admin.ModelAdmin):
    list_display = ('experiment', 'room', 'status', 'start_time')
    list_filter = ('experiment__name', 'status')

class EpisodeAdmin(admin.ModelAdmin):
    list_display = ('session', 'started_at', 'ended_at', 'completed', 'duration_steps')
    list_filter = ('session__id',)

class RecordAdmin(admin.ModelAdmin):
    list_display = ('episode', 'step_index', 'timestamp')
    list_filter = ('episode__id',)

admin.site.register(Session, SessionAdmin)
admin.site.register(Episode, EpisodeAdmin)
admin.site.register(Record, RecordAdmin)
