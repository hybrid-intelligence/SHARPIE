from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    # This is the minimum information we need for an experiment
    goal_velocity = forms.FloatField(
        label='Goal velocity', 
        step_size=0.01, 
        initial=0.0,
        help_text='The target velocity the agent needs to achieve to solve the task.',
        widget=forms.NumberInput(attrs={
            'data-help-text': 'The target velocity the agent needs to achieve to solve the task.',
            'step': '0.01'
        })
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
        initial='For more information, go to the <a href="https://gymnasium.farama.org/environments/classic_control/mountain_car/" target="_blank">Mountain Car page</a>'
    )
