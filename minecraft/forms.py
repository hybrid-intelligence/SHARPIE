from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    initial_prompt = forms.CharField(widget=forms.Textarea(attrs={"rows":"3"}))

    room_name = forms.CharField(label='Room name', max_length=10)
    train = forms.ChoiceField(label='Agent behavior', choices=((True, 'Train'), (False, 'Evaluate')), 
                                 help_text='For more information, go to the <a href="https://github.com/Shalev-Lifshitz/STEVE-1" target="_blank">STEVE-1 page</a>')



class RunForm(forms.Form):
    prompt = forms.CharField(widget=forms.Textarea(attrs={"rows":"3"}))