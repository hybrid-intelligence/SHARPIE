from django import forms

# For example, look at https://docs.djangoproject.com/en/5.1/ref/forms/fields/
class LoginForm(forms.Form):
    # This is the minimum information we need for an experiment
    username = forms.CharField(label='Username', max_length=255)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

class RegisterForm(forms.Form):
    username = forms.CharField(label='Username', max_length=255)
    email = forms.EmailField(label='Email')
    first_name = forms.CharField(label='First name', max_length=255)
    last_name = forms.CharField(label='Last name', max_length=255)
    password1 = forms.CharField(widget=forms.PasswordInput, label='Password')
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirm Password')

class ConsentForm(forms.Form):
    """Form for informed consent agreement."""
    agree = forms.BooleanField(
        required=True,
        label='I agree to participate in the research project as described above'
    )