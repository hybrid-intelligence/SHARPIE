from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    initial_prompt = forms.CharField(
        label='Initial Prompt',
        widget=forms.Textarea(attrs={
            "rows": "3",
            'data-help-text': 'The initial prompt that describes what you want the agent to do in Minecraft.'
        }),
        help_text='The initial prompt that describes what you want the agent to do in Minecraft.'
    )

    room_name = forms.CharField(
        label='Room name', 
        max_length=10,
        help_text='A unique identifier for this experiment session.',
        widget=forms.TextInput(attrs={'data-help-text': 'A unique identifier for this experiment session.'})
    )
    
    train = forms.ChoiceField(
        label='Agent behavior', 
        choices=((True, 'Train'), (False, 'Evaluate')), 
        help_text='Choose "Train" to let the agent learn from experience, or "Evaluate" to test its performance.',
        widget=forms.Select(attrs={'data-help-text': 'Choose "Train" to let the agent learn from experience, or "Evaluate" to test its performance.'})
    )

    # Add a hidden field for documentation link
    doc_link = forms.CharField(
        widget=forms.HiddenInput(),
        initial='For more information, go to the <a href="https://github.com/Shalev-Lifshitz/STEVE-1" target="_blank">STEVE-1 page</a>'
    )


class RunForm(forms.Form):
    prompt = forms.CharField(
        widget=forms.Textarea(attrs={
            "rows": "3",
            'data-help-text': 'Enter your prompt for the agent to execute in Minecraft.'
        }),
        help_text='Enter your prompt for the agent to execute in Minecraft.'
    )