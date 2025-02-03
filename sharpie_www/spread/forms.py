from django import forms

class ConfigForm(forms.Form):
    # For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
    agent_number = forms.IntegerField(label='Number of agents', initial=3)
    local_ratio = forms.FloatField(label='Local ratio', step_size=0.01, initial=0.5)
    max_cycles = forms.IntegerField(label='Maximum number of cycles', initial=25)
    continuous_actions = forms.ChoiceField(label='Environment type', choices=((True, 'Continuous'), (False, 'Discrete')))

    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')))
    agent_file = forms.CharField(label='Agent file', max_length=10, required=False, 
                                 help_text='For more information, go to the <a href="https://pettingzoo.farama.org/environments/mpe/simple_spread/" target="_blank">Simple Spread page</a>')
