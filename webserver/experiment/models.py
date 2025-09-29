from django.db.models.functions import Now
from django.db import models

class Experiment(models.Model):
    name = models.CharField('Name', max_length=100)
    description = models.TextField('Description', blank=True)
    input_list = models.JSONField('Inputs captured from the users', default=list)
    agent_list = models.JSONField('Agents available to play', default=list(['agent_0', 'Agent']))
    users_needed = models.IntegerField('Number of users needed to start the experiment', default=1)
    link = models.CharField('Link to the experiment', max_length=100, blank=True)

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

class Runner(models.Model):
    created_at = models.DateTimeField('Created at', db_default=Now())
    last_active = models.DateTimeField('Last active', db_default=Now())
    status = models.CharField('Status', max_length=20, default='idle')
    current_room = models.CharField('Current room being managed', max_length=20, null=True, blank=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, null=True, blank=True)
    ip_address = models.GenericIPAddressField('IP Address', null=True, blank=True)

    def __str__(self):
        return f"Runner {self.id} - {self.experiment.name if self.experiment else 'No Experiment'}"
    
class Queue(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    room_name = models.CharField('Room name', max_length=20)
    users_waiting = models.IntegerField('Number of users currently waiting', default=0) 
    status = models.CharField('Status of the queue', max_length=20, default='waiting')  # e.g., waiting, running, terminated, dead
    created_at = models.DateTimeField('Created at', db_default=Now())

    def __str__(self):
        return f"{self.experiment.name} - {self.room_name} ({self.users_waiting} users waiting)"