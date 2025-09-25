from django.db.models.functions import Now
from django.db import models

class Experiment(models.Model):
    name = models.CharField('Name', max_length=100)
    description = models.TextField('Description', blank=True)
    input_list = models.JSONField('Inputs captured from the users', default=list)
    user_number = models.IntegerField('Number of users in the same room', default=1)

    def __str__(self):
        return self.name

class Trial(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    room_name = models.CharField('Room name', max_length=20)
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, null=True, blank=True)
    agent_played = models.CharField('Agent played by the user', max_length=50)
    started_at = models.DateTimeField('Started at', db_default=Now())
    ended_at = models.DateTimeField('Ended at', null=True, blank=True)

    def __str__(self):
        return f"{self.experiment.name} - {self.room_name}"

class Interaction(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(default=Now())
    step = models.IntegerField('Step number in the episode', default=0)
    observations = models.JSONField('Observation from the environment', default=dict)
    actions = models.JSONField('Action taken by the user', default=dict)
    rewards = models.JSONField('Reward received from the environment', default=dict)