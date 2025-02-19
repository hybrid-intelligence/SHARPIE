from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class ConfigForm(forms.Form):
    # This is the minimum information we need for an experiment
    room_name = forms.CharField(label='Room name', max_length=10, 
                                help_text='For more information, go to the <a href="https://github.com/libgoncalv/SHARPIE" target="_blank">SHARPIE GitHub</a>')
