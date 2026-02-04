from django.db.models.functions import Now
from django.db import models
from django.core.exceptions import ValidationError

class Experiment(models.Model):
    name = models.CharField('Name', max_length=100)
    type = models.CharField('Type ("action" or "reward")', max_length=50, default='action')
    description = models.TextField('Description', blank=True)
    input_list = models.JSONField('Inputs captured from the users', default=list)
    agent_list = models.JSONField('Agents available to play', default=[['agent_0', 'Agent']])
    users_needed = models.IntegerField('Number of users needed to start the experiment', default=1)
    link = models.CharField('Link to the experiment', max_length=100, unique=True, help_text="Unique identifier (slug) for the experiment URL")
    episodes_to_complete = models.IntegerField('Number of episodes to complete per user', default=1)
    target_fps = models.FloatField('Target FPS for the experiment', default=24.0)
    train = models.BooleanField('Whether the agent is trained during the experiment', default=False)

    def clean(self):
        # Validate input_list: must be a list of strings
        if not isinstance(self.input_list, list):
            raise ValidationError({'input_list': 'Input list must be a list.'})
        if not all(isinstance(item, str) for item in self.input_list):
            raise ValidationError({'input_list': 'All items in input list must be strings (keyboard keys).'})
        
        # Validate agent_list: must be a list of lists/tuples with at least one element
        if not isinstance(self.agent_list, list):
            raise ValidationError({'agent_list': 'Agent list must be a list.'})
        if len(self.agent_list) == 0:
            raise ValidationError({'agent_list': 'Agent list cannot be empty.'})
        for item in self.agent_list:
            if not isinstance(item, (list, tuple)):
                raise ValidationError({'agent_list': 'Each agent must be a list or tuple.'})
            if len(item) == 0:
                raise ValidationError({'agent_list': 'Each agent item cannot be empty.'})
            if not isinstance(item[0], str):
                raise ValidationError({'agent_list': 'Agent value must be a string.'})
    
    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

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
    evaluate = models.BooleanField('Whether the experiment is in evaluation mode', default=False)
    users_waiting = models.IntegerField('Number of users currently waiting', default=0) 
    status = models.CharField('Status of the queue', max_length=20, default='waiting')  # e.g., waiting, running, terminated, dead
    created_at = models.DateTimeField('Created at', db_default=Now())

    def __str__(self):
        return f"{self.experiment.name} - {self.room_name} ({self.users_waiting} users waiting)"