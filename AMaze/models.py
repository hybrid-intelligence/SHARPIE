from django.db import models
from django.contrib.auth.models import User

class Info(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='AMaze')
    room = models.CharField(max_length=60) 
    step = models.IntegerField(default=-1) 
    reward = models.FloatField(default=0.0) 
    changed = models.BooleanField(default=False) 