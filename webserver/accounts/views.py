from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from .forms import LoginForm, RegisterForm


def login_(request):
    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = LoginForm(request.POST)
        # Check whether it's valid
        if form.is_valid():
            user = authenticate(request, username=form.cleaned_data['username'], password=form.cleaned_data['password'])
            if user is not None:
                login(request, user)

    if request.user.is_authenticated:
        return redirect(request.GET.get('next', '/'))
    
    # Create empty config form
    form = LoginForm()
    return render(request, "registration/login.html", {'form': form, 'view': 'login'})

def register_(request):
    # Create empty register form
    form = RegisterForm()

    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = RegisterForm(request.POST)
    
    
    error = None
    # If GET request and info are provided, prefill the form
    if request.method == "GET" and 'username' in request.GET and 'study' in request.GET and 'registration_key' in request.GET:
        form = RegisterForm({'username': request.GET.get('username', ''),
                            'first_name': "Participant "+request.GET.get('username', ''),
                            'last_name': "Study "+request.GET.get('study', ''),
                            'password1': request.GET.get('username', '')+'_'+request.GET.get('study', ''),
                            'password2': request.GET.get('username', '')+'_'+request.GET.get('study', ''),
                            'registration_key': request.GET.get('registration_key', '')})
    
    if form.is_valid() and form.cleaned_data['password1']==form.cleaned_data['password2']:
        try:
            User.objects.get(username=form.cleaned_data['username'])
            error = "Username already exists. Please choose another or login."
        except User.DoesNotExist:
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'] if 'email' in form.cleaned_data else 'john.doe@example.com',
                password=form.cleaned_data['password1'],
                last_name=form.cleaned_data['last_name'] if 'last_name' in form.cleaned_data else '',
                first_name=form.cleaned_data['first_name'] if 'first_name' in form.cleaned_data else ''
            )
            if user is not None:
                login(request, user)
    elif form.is_valid() and form.cleaned_data['password1']!=form.cleaned_data['password2']:
        error = "Passwords do not match."
        
    if request.user.is_authenticated:
        return redirect(request.GET.get('next', '/'))

    return render(request, "registration/login.html", {'form': form, 'view': 'register', 'error': error})

def logout_(request):
    if request.user.is_authenticated:
        logout(request)
    
    return redirect('/')