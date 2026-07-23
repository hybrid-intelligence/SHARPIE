"""Form for configuring a participant's experiment session."""
from django import forms
import re

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    """Form for a participant to configure their room and role before joining an experiment session."""
    room = forms.CharField(
        label='Room name', 
        max_length=20,
        help_text='A unique identifier for this experiment session.',
        widget=forms.TextInput(attrs={'data-help-text': 'A unique identifier for this experiment session.'})
    )
    
    role = forms.ChoiceField(
        label='Role', 
        choices = [['agent_0', 'Agent']], # Default choice, will be updated in the view
        help_text='Which role you want to play',
    )

    # Add a hidden field for documentation link
    doc_link = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        initial='For more information, go to the <a href="https://github.com/libgoncalv/SHARPIE" target="_blank">SHARPIE GitHub</a>'
    )

    def clean(self):
        """Validate that the room name can be used in URLS, contains only alphanumerics, hyphens, underscores, or periods."""
        cleaned_data = super().clean()
        # Check if the room is valid unicode string with length < 100 containing only ASCII alphanumerics, hyphens, underscores, or periods
        if not re.match(r'^[A-Za-z0-9-_\.]+$', cleaned_data['room']):
            raise forms.ValidationError("Room must be a unique identifier with only alphanumeric characters, hyphens, underscores, or periods.")