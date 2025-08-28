from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    num_good = forms.IntegerField(
        label='Number of good agents', 
        initial=1,
        help_text='The number of good agents in the environment.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The number of good agents in the environment.'})
    )
    
    num_adversaries = forms.IntegerField(
        label='Number of adversaries', 
        initial=3,
        help_text='The number of adversary agents in the environment.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The number of adversary agents in the environment.'})
    )
    
    num_obstacles = forms.IntegerField(
        label='Number of obstacles', 
        initial=2,
        help_text='The number of obstacles in the environment.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The number of obstacles in the environment.'})
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
    
    AGENT_CHOICES = [
        ('adversary_0', 'adversary_0'),
        ('adversary_1', 'adversary_1'),
        ('adversary_2', 'adversary_2'),
        ('agent_0', 'agent_0'),
    ]
    played_agent = forms.ChoiceField(
        label='Played agent',
        choices=AGENT_CHOICES,
        help_text='Which agent you want to play (i.e. adversary_0, adversary_1, adversary_2, agent_0, etc)',
        widget=forms.Select(attrs={'data-help-text': 'Which agent you want to play (i.e. adversary_0, adversary_1, adversary_2, agent_0, etc)'})
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
        initial='For more information, go to the <a href="https://pettingzoo.farama.org/environments/mpe/simple_tag" target="_blank">Simple Tag page</a>'
    )
