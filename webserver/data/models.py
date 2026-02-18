from django.db.models.functions import Now
from django.db import models
from django.core.exceptions import ValidationError

from experiment.models import Experiment, Participant



class Session(models.Model):
    """
    The run of a single participant or group of participants through an experiment. Consists of one or more episodes.
    """
    experiment = models.ForeignKey(Experiment, on_delete=models.DO_NOTHING, related_name='sessions')
    participants = models.ManyToManyField(Participant, related_name='sessions')
    room = models.CharField(max_length=100)
    connected_participants = models.IntegerField(default=0)

    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True)

    status = models.CharField(max_length=15, choices=[('not_ready', 'Not ready'), ('ready', 'Ready'), ('pending', 'Pending'), ('running', 'Running'), ('completed', 'Completed'), ('aborted', 'Aborted')], default='not_ready')
    metadata = models.JSONField(null=True, blank=True)

    # Check if that session is unique for the experiment and room combination. This is used to prevent multiple sessions from being created with the same experiment and room combination.
    def validate_unique_running_session(self):
        return not Session.objects.filter(experiment=self.experiment, room=self.room, status__in=['ready', 'pending', 'running']).exists()

class Episode(models.Model):
    """
    A complete sequence of interactions between a set of agents, and their environment—from an initial state to a terminal or truncated state—representing one coherent task of decision-making cycle.
    """
    session = models.ForeignKey(Session, on_delete=models.DO_NOTHING, related_name='episodes')

    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True)

    duration_steps = models.IntegerField(null=True)
    completed = models.BooleanField(default=False)
    outcome = models.JSONField(null=True, blank=True)

class Record(models.Model):
    """
    A single entry within that captures the state of the environment, an agent’s (human or artificial) actions or other outputs, the resulting outcomes or rewards, and any relevant contextual information at a specific point during a trial.
    """
    episode = models.ForeignKey(Episode, on_delete=models.DO_NOTHING, related_name='records')
    step_index = models.IntegerField()
    state = models.JSONField(null=True, blank=True)
    action = models.JSONField(null=True, blank=True)
    reward = models.JSONField(null=True, blank=True)
    info = models.JSONField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
