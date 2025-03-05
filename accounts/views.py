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
    return render(request, "registration/login.html", {'form': form})

def register_(request):
    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = RegisterForm(request.POST)
        # Check whether it's valid
        if form.is_valid() and form.cleaned_data['password1']==form.cleaned_data['password2']:
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password1'],
                last_name=form.cleaned_data['last_name'],
                first_name=form.cleaned_data['first_name']
            )
            if user is not None:
                login(request, user)

    if request.user.is_authenticated:
        return redirect(request.GET.get('next', '/'))
    
    # Create empty config form
    form = RegisterForm()
    return render(request, "registration/login.html", {'form': form})

def logout_(request):
    if request.user.is_authenticated:
        logout(request)
    
    return redirect('/')