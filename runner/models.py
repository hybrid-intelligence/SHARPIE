from django.db import models

class Experiment(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    # other fields...

    def __str__(self):
        return self.name