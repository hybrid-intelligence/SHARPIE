from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.utils import timezone
from .forms import LoginForm, RegisterForm, ConsentForm, ProfileInfoForm, ProfilePasswordForm
from .models import Consent


def has_consented(user):
    """Check if user has given consent."""
    if not user.is_authenticated:
        return False
    try:
        consent = Consent.objects.get(user=user)
        return consent.agreed
    except Consent.DoesNotExist:
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
        # Check if user has consented, if not redirect to consent page
        if not has_consented(request.user):
            return redirect('/accounts/consent/')
        return redirect(request.GET.get('next', '/'))
    
    # Create empty config form
    form = RegisterForm()
    return render(request, "registration/login.html", {'form': form})

def logout_(request):
    if request.user.is_authenticated:
        logout(request)
    
    return redirect('/')


@login_required
def consent_(request):
    """Handle informed consent form."""
    # Check if user already has consent
    try:
        consent = Consent.objects.get(user=request.user)
    except Consent.DoesNotExist:
        consent = None
    
    if request.method == "POST":
        form = ConsentForm(request.POST)
        if form.is_valid() and form.cleaned_data['agree']:
            # Get or create consent record
            consent, created = Consent.objects.get_or_create(user=request.user)
            consent.agreed = True
            consent.agreed_at = timezone.now()
            # Try to get IP address
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip_address = x_forwarded_for.split(',')[0]
            else:
                ip_address = request.META.get('REMOTE_ADDR')
            consent.ip_address = ip_address
            consent.save()
            # Redirect to the next page or home
            return redirect(request.GET.get('next', '/'))
    else:
        form = ConsentForm()
    
    return render(request, "registration/consent.html", {'form': form, 'consent': consent})


@login_required
def profile_(request):
    formInfo = ProfileInfoForm(initial={'email': request.user.email,
                                        'first_name': request.user.first_name,
                                        'last_name': request.user.last_name,}) 
    formPassword = ProfilePasswordForm()

    if request.method == "POST":
        formInfo = ProfileInfoForm(request.POST)
        if formInfo.is_valid():
            request.user.email = formInfo.cleaned_data['email']
            request.user.first_name = formInfo.cleaned_data['first_name']
            request.user.last_name = formInfo.cleaned_data['last_name']
            request.user.save()
        formPassword = ProfilePasswordForm(request.POST)
        if formPassword.is_valid() and formPassword.cleaned_data['password1'] == formPassword.cleaned_data['password2']:
            request.user.set_password(formPassword.cleaned_data['password'])
            request.user.save()
        return redirect('/accounts/profile/')
    
    try:
        consent = Consent.objects.get(user=request.user)
    except Consent.DoesNotExist:
        consent = None

    return render(request, "registration/profile.html", {'formInfo': formInfo, 'formPassword': formPassword, 'consent': consent})