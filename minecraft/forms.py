from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')), 
                                 help_text='For more information, go to the <a href="https://minerl.readthedocs.io/en/latest/index.html" target="_blank">MineRL page</a>')
