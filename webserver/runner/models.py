from django.db.models.functions import Now
from django.db import models
from django.core.exceptions import ValidationError

from data.models import Session



class Runner(models.Model):
    """
    Component that manages the execution of experiments.
    """
    connection_key = models.CharField(max_length=50, unique=True)
    status = models.CharField('Status', max_length=20, default='never connected')
    last_active = models.DateTimeField('Last active', db_default=Now(), null=True, blank=True)
    session = models.ForeignKey(Session, on_delete=models.DO_NOTHING, null=True, blank=True)
    ip_address = models.GenericIPAddressField('IP Address', null=True, blank=True)

    def __str__(self):
        return f"Runner {self.id} - {self.status}"
