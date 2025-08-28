from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    environment = forms.ChoiceField(
        label='Environment name', 
        choices=(('PandaReachDense-v3', 'PandaReachDense-v3'),
                  ('PandaPushDense-v3', 'PandaPushDense-v3'),
                  ('PandaSlideDense-v3', 'PandaSlideDense-v3'),
                  ('PandaPickAndPlaceDense-v3', 'PandaPickAndPlaceDense-v3'),
                  ('PandaStackDense-v3', 'PandaStackDense-v3'),
                  ('PandaFlipDense-v3', 'PandaFlipDense-v3')),
        help_text='Choose between different tasks.',
        widget=forms.Select(attrs={'data-help-text': 'Choose between different environments.'})
    )

    algorithm_name = forms.ChoiceField(
        label='Algorithm name', 
        choices=(('DDPG', 'DDPG'), ('SAC','SAC')),
        help_text='Choose the RLmax_episode_steps algorithm.',
        widget=forms.Select(attrs={'data-help-text': 'Choose between different algorithms.'})
    )

    max_episode_steps = forms.IntegerField(
        label='Maximum number of episode steps', 
        initial=500,
        help_text='The maximum number of steps in a single episode.',
        widget=forms.NumberInput(attrs={'data-help-text': 'The maximum number of cycles before the episode ends.'})
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
