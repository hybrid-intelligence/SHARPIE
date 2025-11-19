from django.db import models
from django.contrib.auth.models import User


class Consent(models.Model):
    """Model to track user consent for research participation."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='consent')
    agreed = models.BooleanField(default=False)
    agreed_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    def __str__(self):
        return f"Consent for {self.user.username} - {'Agreed' if self.agreed else 'Not Agreed'}"

