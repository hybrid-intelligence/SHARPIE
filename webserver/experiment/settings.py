from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    # This is the minimum information we need for an experiment
    room_name = forms.CharField(
        label='Room name', 
        max_length=20,
        help_text='A unique identifier for this experiment session.',
        widget=forms.TextInput(attrs={'data-help-text': 'A unique identifier for this experiment session.'})
    )
    
    agent = forms.CharField(
        required=False,
        label='Played agent', 
        initial='agent_0',
        help_text='Which agent you want to play',
        widget=forms.HiddenInput(),
    )

    # Add a hidden field for documentation link
    doc_link = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        initial='For more information, go to the <a href="https://github.com/libgoncalv/SHARPIE" target="_blank">SHARPIE GitHub</a>'
    )

experiment_name = "Mountain car"
inputsListened = ['ArrowLeft', 'ArrowDown', 'ArrowRight']