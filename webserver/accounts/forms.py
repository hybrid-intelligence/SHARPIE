from django import forms
from mysite.settings import REGISTRATION_KEY
from django.core.validators import RegexValidator

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class LoginForm(forms.Form):
    # This is the minimum information we need for an experiment
    username = forms.CharField(label='Username', max_length=255)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

class RegisterForm(forms.Form):
    username = forms.CharField(label='Username', max_length=255, help_text='255 characters or fewer. Letters, digits and @/./+/-/_ only.')
    first_name = forms.CharField(label='First name', max_length=255, required=False, 
                                 widget=forms.TextInput(attrs={'placeholder': 'John'}))
    last_name = forms.CharField(label='Last name', max_length=255, required=False,
                                widget=forms.TextInput(attrs={'placeholder': 'Doe'}))
    email = forms.EmailField(label='Email', required=False, 
                             widget=forms.TextInput(attrs={'placeholder': 'john.doe@example.com'}))
    password1 = forms.CharField(widget=forms.PasswordInput, label='Password')
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirm Password',
                                help_text='Enter the same password as before, for verification.')

    # Only add the registration key field if a registration key is set
    if REGISTRATION_KEY and isinstance(REGISTRATION_KEY, str):
        registration_key = forms.CharField(label='Registration Key', max_length=255, 
                                           validators=[RegexValidator(regex='^'+REGISTRATION_KEY+'$', message='Invalid registration key.')],
                                           widget=forms.TextInput(attrs={'placeholder': 'Enter registration key'}))

class ConsentForm(forms.Form):
    """Form for informed consent agreement."""
    agree = forms.BooleanField(
        required=True,
        label='I agree to participate in the research project as described above'
    )

class ProfileInfoForm(forms.Form):
    """Form for updating user profile information."""
    username = forms.CharField(label='Username', max_length=255, required=True)
    email = forms.EmailField(label='Email', required=False,
                             widget=forms.TextInput(attrs={'placeholder': 'You can add your email here'}))
    first_name = forms.CharField(label='First name', max_length=255, required=False,
                             widget=forms.TextInput(attrs={'placeholder': 'You can add your first name here'}))
    last_name = forms.CharField(label='Last name', max_length=255, required=False,
                             widget=forms.TextInput(attrs={'placeholder': 'You can add your last name here'}))

class ProfilePasswordForm(forms.Form):
    password1 = forms.CharField(widget=forms.PasswordInput, label='Password')
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirm Password',
                                help_text='Enter the same password as before, for verification.')
