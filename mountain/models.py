from django.db import models
from django.contrib.auth.models import User

class Info(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mountain')
    room = models.CharField(max_length=60) 
    step = models.IntegerField(default=-1) 
    action = models.IntegerField(default=-1) 