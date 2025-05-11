from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    agent_number = forms.IntegerField(
        label='Number of agents', 
        initial=3,
        help_text='The number of agents that will interact in the environment.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The number of agents that will interact in the environment.'})
    )
    
    local_ratio = forms.FloatField(
        label='Local ratio', 
        step_size=0.01, 
        initial=0.5,
        help_text='The ratio of local observations to global observations.',
        widget=forms.NumberInput(attrs={
            'data-help-text': 'The ratio of local observations to global observations.',
            'step': '0.01'
        })
    )
    
    max_cycles = forms.IntegerField(
        label='Maximum number of cycles', 
        initial=25,
        help_text='The maximum number of cycles before the episode ends.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The maximum number of cycles before the episode ends.'})
    )
    
    continuous_actions = forms.ChoiceField(
        label='Environment type', 
        choices=((True, 'Continuous'), (False, 'Discrete')),
        help_text='Choose between continuous or discrete action spaces.',
        widget=forms.Select(attrs={'data-help-text': 'Choose between continuous or discrete action spaces.'})
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
        initial='For more information, go to the <a href="https://pettingzoo.farama.org/environments/mpe/simple_spread/" target="_blank">Simple Spread page</a>'
    )
