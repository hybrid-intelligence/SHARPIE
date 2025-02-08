from django import forms


class ConfigForm(forms.Form):
    # For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')))
    agent_file = forms.CharField(label='Agent file', max_length=10, required=False, 
                                 help_text='For more information, go to the <a href="https://gymnasium.farama.org/environments/classic_control/mountain_car/" target="_blank">Mountain Car page</a>')
