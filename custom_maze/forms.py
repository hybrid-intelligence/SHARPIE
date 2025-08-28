from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    environment = forms.ChoiceField(
        label='Environment', 
        choices=(('pygame', 'Grid-Pygame'),
                  ('Frozen Lake','Frozen Lake'),
                  ('PointMaze_UMazeDense-v3', 'UMazeDense'),
                  ('PointMaze_OpenDense-v3', 'OpenDense'),
                  ('PointMaze_MediumDense-v3', 'MediumDense'),
                  ('PointMaze_LargeDense-v3', 'LargeDense')),
        help_text='Choose between different environments.',
        widget=forms.Select(attrs={'data-help-text': 'Choose between different environments.'})
    )



    algorithm_name = forms.ChoiceField(
        label='Algorithm name', 
        choices=(('DQN', 'DQN'), ('Q-table','Q-table')),
        help_text='Choose the RL algorithm.',
        widget=forms.Select(attrs={'data-help-text': 'Choose between different algorithms.'})
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