from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    # This is the minimum information we need for an experiment
    seed = forms.IntegerField(
        label='Random seed', 
        initial=12, 
        min_value=1, 
        max_value=10000,
        help_text='A number used to initialize the random number generator. Using the same seed will produce the same maze layout.',
        widget=forms.NumberInput(attrs={'data-help-text': 'A number used to initialize the random number generator. Using the same seed will produce the same maze layout.'})
    )
    width = forms.IntegerField(
        label='Width', 
        initial=10, 
        min_value=5, 
        max_value=30,
        help_text='The width of the maze in cells. Must be between 5 and 30.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The width of the maze in cells. Must be between 5 and 30.'})
    )
    height = forms.IntegerField(
        label='Height', 
        initial=10, 
        min_value=5, 
        max_value=30,
        help_text='The height of the maze in cells. Must be between 5 and 30.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The height of the maze in cells. Must be between 5 and 30.'})
    )
    unicursive = forms.ChoiceField(
        label='Unicursive maze', 
        choices=((True, 'Yes'), (False, 'No')),
        help_text='If enabled, the maze will have only one solution path from start to goal.',
        widget=forms.Select(attrs={'data-help-text': 'If enabled, the maze will have only one solution path from start to goal.'})
    )
    lures = forms.IntegerField(
        label='Lures', 
        initial=0, 
        min_value=0, 
        max_value=30,
        help_text='Number of lure cells that provide positive rewards but are not on the optimal path.',
        widget=forms.NumberInput(attrs={'data-help-text': 'Number of lure cells that provide positive rewards but are not on the optimal path.'})
    )
    traps = forms.IntegerField(
        label='Traps', 
        initial=0, 
        min_value=0, 
        max_value=30,
        help_text='Number of trap cells that provide negative rewards.',
        widget=forms.NumberInput(attrs={'data-help-text': 'Number of trap cells that provide negative rewards.'})
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
        initial='For more information, go to the <a href="https://amaze.readthedocs.io/en/latest/" target="_blank">AMaze page</a>'
    )


class RunForm(forms.Form):
    # For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
    reward = forms.FloatField(
        label='Reward',
        help_text='The reward value for the current state.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The reward value for the current state.'})
    )