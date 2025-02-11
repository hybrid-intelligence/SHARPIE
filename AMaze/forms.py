from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    # This is the minimum information we need for an experiment
    seed = forms.IntegerField(label='Random seed', initial=12, min_value=1, max_value=10000)
    width = forms.IntegerField(label='Width', initial=10, min_value=5, max_value=30)
    height = forms.IntegerField(label='Height', initial=10, min_value=5, max_value=30)
    unicursive = forms.ChoiceField(label='Unicursive maze', choices=((True, 'Yes'), (False, 'No')))
    lures = forms.IntegerField(label='Lures', initial=0, min_value=0, max_value=30)
    traps = forms.IntegerField(label='Traps', initial=0, min_value=0, max_value=30)

    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')), 
                                 help_text='For more information, go to the <a href="https://amaze.readthedocs.io/en/latest/" target="_blank">AMaze page</a>')


class RunForm(forms.Form):
    # For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
    reward = forms.FloatField(label='Reward')