from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.utils import timezone

from mysite.settings import REGISTRATION_KEY, DEMO
from .forms import LoginForm, RegisterForm, ConsentForm, ProfileInfoForm, ProfilePasswordForm
from .models import Consent, Participant


def has_consented(user):
    """Check if user has given consent."""
    if not user.is_authenticated:
        return False
    try:
        participant = Participant.objects.get(user=user)
        return participant.agreed_at and not participant.withdrawn_at
    except Participant.DoesNotExist:
        participant = Participant(user=user)
        participant.save()
        return False


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
        # Check if user has consented, if not redirect to consent page
        if not has_consented(request.user):
            return redirect('/accounts/consent/')
        return redirect(request.GET.get('next', '/'))
    
    # Create empty config form
    form = LoginForm()
    return render(request, "registration/login.html", {'form': form, 'view': 'login'})

def register_(request):
    # If registration is disabled
    if REGISTRATION_KEY == False:
        return render(request, "registration/login.html", {'form': None, 'view': 'register', 'error': None})

    # Create empty register form
    form = RegisterForm()

    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = RegisterForm(request.POST)
    
    
    error = None
    external_id = None
    # If GET request and info are provided, prefill the form
    if request.method == "GET" and 'username' in request.GET and 'study' in request.GET and 'registration_key' in request.GET:
        form = RegisterForm({'username': request.GET.get('username', ''),
                            'first_name': "Participant "+request.GET.get('username', ''),
                            'last_name': "Study "+request.GET.get('study', ''),
                            'password1': request.GET.get('username', '')+'_'+request.GET.get('study', ''),
                            'password2': request.GET.get('username', '')+'_'+request.GET.get('study', ''),
                            'registration_key': request.GET.get('registration_key', '')})
        external_id = request.GET.get('username', '')
    
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
                participant = Participant(user=user, external_id=external_id)
                participant.save()
                login(request, user)

    elif form.is_valid() and form.cleaned_data['password1']!=form.cleaned_data['password2']:
        error = "Passwords do not match."
        
    if request.user.is_authenticated:
        # Check if user has consented, if not redirect to consent page
        if not has_consented(request.user):
            return redirect('/accounts/consent/')
        return redirect(request.GET.get('next', '/'))

    return render(request, "registration/login.html", {'form': form, 'view': 'register', 'error': error, 'DEMO': DEMO, 'REGISTRATION_KEY': REGISTRATION_KEY})

def logout_(request):
    if request.user.is_authenticated:
        logout(request)
    
    return redirect('/')


@login_required
def consent_(request):
    """Handle informed consent form."""
    # Get the participant, otherwise create it
    try:
        participant = Participant.objects.get(user=request.user)
        consent = participant.consent
    except Participant.DoesNotExist:
        participant = Participant(user=request.user)
        participant.save()

    if not consent:
        consent = Consent.objects.last()

    if request.method == "POST":
        # Handle consent retraction
        if 'retract_consent' in request.POST:
            participant.withdrawn_at = timezone.now()
            participant.save()
            return redirect('/accounts/consent/')
        # Handle consent agreement
        form = ConsentForm(request.POST)
        if form.is_valid() and form.cleaned_data['agree']:
            participant.consent = consent
            participant.agreed_at = timezone.now()
            participant.save()
            # Redirect to the next page or home
            return redirect(request.GET.get('next', '/accounts/consent/'))
    else:
        form = ConsentForm()
    
    return render(request, "registration/consent.html", {'form': form, 'participant': participant, 'consent': consent})


@login_required
def profile_(request):
    # Get the participant, otherwise create it
    try:
        participant = Participant.objects.get(user=request.user)
    except Participant.DoesNotExist:
        participant = Participant(user=request.user)
        participant.save()
    
    disabled = (not participant.agreed_at or participant.withdrawn_at or participant.external_id)
    formInfo = ProfileInfoForm(disabled, initial={'username': request.user.username,
                                        'email': request.user.email,
                                        'first_name': request.user.first_name,
                                        'last_name': request.user.last_name,}) 
    formPassword = ProfilePasswordForm(disabled)

    if request.method == "POST":
        # Handle profile info update
        if 'update_info' in request.POST:
            formInfo = ProfileInfoForm(disabled, request.POST)
            if formInfo.is_valid():
                # Check if username is being changed and if it's available
                new_username = formInfo.cleaned_data['username']
                username_valid = True
                if new_username != request.user.username:
                    # Check if username is already taken
                    if User.objects.filter(username=new_username).exclude(pk=request.user.pk).exists():
                        formInfo.add_error('username', 'This username is already taken.')
                        username_valid = False
                
                # Only update if username is valid (either unchanged or available)
                if username_valid:
                    if new_username != request.user.username:
                        request.user.username = new_username
                    request.user.email = formInfo.cleaned_data['email']
                    request.user.first_name = formInfo.cleaned_data['first_name']
                    request.user.last_name = formInfo.cleaned_data['last_name']
                    request.user.save()
                    return redirect('/accounts/profile/')
        
        # Handle password update
        if 'update_password' in request.POST:
            formPassword = ProfilePasswordForm(disabled, request.POST)
            if formPassword.is_valid():
                if formPassword.cleaned_data['password1'] == formPassword.cleaned_data['password2']:
                    request.user.set_password(formPassword.cleaned_data['password1'])
                    request.user.save()
                    return redirect('/accounts/profile/')
                else:
                    formPassword.add_error('password2', 'Passwords do not match.')

    return render(request, "registration/profile.html", {'formInfo': formInfo, 'formPassword': formPassword, 'participant': participant})