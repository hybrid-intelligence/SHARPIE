from django import forms

class ConfigForm(forms.Form):
    # For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
    num_good = forms.IntegerField(label='Number of good agents', initial=1)
    num_adversaries = forms.IntegerField(label='Number of adversaries', initial=3)
    num_obstacles = forms.IntegerField(label='Number of obstacles', initial=2)
    max_cycles = forms.IntegerField(label='Maximum number of cycles', initial=25)
    continuous_actions = forms.ChoiceField(label='Environment type', choices=((True, 'Continuous'), (False, 'Discrete')))
    played_agent = forms.CharField(label='Played agent', max_length=20, help_text='Which agent you want to play (i.e. adversary_0, adversary_1, adversary_2, agent_0, etc)')

    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')))
    agent_file = forms.CharField(label='Agent file', max_length=10, required=False, 
                                 help_text='For more information, go to the <a href="https://pettingzoo.farama.org/environments/mpe/simple_tag" target="_blank">Simple Tag page</a>')
