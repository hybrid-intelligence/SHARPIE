from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    # This is the minimum information we need for an experiment
    goal_velocity = forms.FloatField(label='Goal velocity', step_size=0.01, initial=0.0)

    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')), 
                              help_text='For more information, go to the <a href="https://gymnasium.farama.org/environments/classic_control/mountain_car/" target="_blank">Mountain Car page</a>')
