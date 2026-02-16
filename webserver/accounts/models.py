from django.db import models
from django.contrib.auth.models import User


class Consent(models.Model):
    """Model to formulate user consent for research participation."""
    name = models.CharField(max_length=100)
    explanation_text = models.TextField(max_length=1000, default="This study involves participation in interactive experiments using the SHARPIE (Shared Human-AI Reinforcement Learning Platform for Interactive Experiments) platform. You will interact with AI agents in reinforcement learning environments. The experiment typically takes 15-30 minutes to complete, depending on the specific experiment configuration.")
    research_team = models.JSONField(default=lambda: dict(main=dict(designation="Principal investigator",
                                                        institution="Hybrid Intelligence Centre",
                                                        location="The Netherlands",
                                                        email="principal.investigator@example.com")))
    ethical_guidelines = models.TextField(max_length=1000, default="This project adheres to the ethical guidelines established by The Hybrid Intelligence Centre, and falls under fundamental research without any commercial purpose nor external stakeholders or partners. For details of our legal basis for using personal data and the rights you have over your data, please see the privacy information at HI Centre Ethics Policy (PDF).")
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.pk:
            raise ValueError("Cannot change consent description after it has been created")
        return super().save(*args, **kwargs)
    
    def delete(self, using = ..., keep_parents = ...):
        raise ValueError("Cannot delete consent description after it has been created")


class Participant(models.Model):
    """Model to track participants in the study."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='participant')
    external_id = models.CharField(max_length=255, unique=True, null=True, blank=True)
    demographics = models.JSONField(null=True, blank=True)
    
    consent = models.ForeignKey(Consent, null=True, on_delete=models.CASCADE, related_name='participants')
    agreed_at = models.DateTimeField(null=True, blank=True)
    withdrawn_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, null=False)
    last_modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Participant: {self.user.username} - ID: {self.external_id}"

