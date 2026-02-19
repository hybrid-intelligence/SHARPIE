import string
import random
import os

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from accounts.decorators import consent_required
from django.contrib.admin.views.decorators import staff_member_required
from django.http import Http404, HttpResponse
from experiment.forms import ConfigForm
from experiment import models

from accounts.models import Participant
from data.models import Session

from django.conf import settings




# Configuration view that will automatically check and save the parameters into the user session variable
@login_required
@consent_required
def config_(request, link):
    # Check the experiment
    try:
        experiment = models.Experiment.objects.get(link=link)
        if not experiment.enabled:
            raise Http404("Experiment is disabled")
    except models.Experiment.DoesNotExist:
        raise Http404("Experiment does not exist")
    
    experiment_roles = []
    for agent in experiment.agents.filter(participant=True):
        experiment_roles.append((agent.role, agent.name))
    
    error_message = None
    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = ConfigForm(request.POST)
        form.fields['role'].choices = experiment_roles
        # Check whether it's valid
        if form.is_valid():
            # Process the data in form.cleaned_data if the field has been filled
            for k in form.fields.keys():
                if k in form.cleaned_data.keys() and k != 'doc_link':  # Don't save doc_link to session
                    request.session[k] = form.cleaned_data[k]
            
            # Try to find if the participant is currently linked to a session and abort it
            for session in request.user.participant.sessions.all():
                if session.status in ['not_ready', 'ready', 'pending', 'running']:
                    session.status = 'aborted'
                    session.save()
                    request.session['session'] = None

            # Look for an open session or create one
            try:
                session = Session.objects.get(experiment=experiment, room=form.cleaned_data['room'], status='not_ready')
            except Session.DoesNotExist:
                session = Session(experiment=experiment, room=form.cleaned_data['room'])
                # Check that the room is not already in use
                if session.validate_unique_running_session():
                    session.save()
                    session = Session.objects.get(experiment=experiment, room=form.cleaned_data['room'], status='not_ready')
                else:
                    error_message = 'This room is already in use, please chose a different one.'
            if not error_message:
                # Add the participant and check if ready
                session.participants.add(request.user.participant)
                if len(session.participants.all()) == len(experiment_roles):
                    session.status = 'ready'
                session.save()
                # Save the SHARPIE session id to the user's web session
                request.session['session'] = session.id

            # Redirect to the run page if there are no errors
            if not error_message:
                return redirect(f"/experiment/{link}/config")
        else:
            error_message = form.errors.get_json_data()['__all__'][0]['message']

    # Create empty config form with the configuration options from the database
    form = ConfigForm()
    form.fields['role'].choices = experiment_roles
    # Use a random room name if the experiment only requires one participant
    if len(experiment_roles) == 1:
        form.fields['room'].initial = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    # Actively check if the user has a session that is currently running or pending
    if 'session' in request.session.keys():
        try:
            session = Session.objects.get(id=request.session['session'], status__in=['not_ready', 'ready', 'pending', 'running'])
        except Session.DoesNotExist:
            request.session['session'] = None

    # Check if all the required fields have been saved in the user session variable
    saved = all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was already saved by the user, we create a prefilled form
    if(saved):
        initial_data = {k: request.session.get(k, None) for k in form.fields.keys() if k != 'doc_link'}
        form = ConfigForm(initial=initial_data)
        form.fields['role'].choices = experiment_roles

    return render(request, "experiment/config.html", {'form': form, 'error_message': error_message, 'saved': saved, 'experiment': experiment})




@login_required
@consent_required
def run_(request, link, room):
    # Get a matching session and check if the participant is linked to it
    try:
        session = Session.objects.get(experiment__link=link, room=room, status__in=['not_ready', 'ready', 'pending', 'running'])
        if not session.participants.filter(id=request.user.participant.id).exists():
            return redirect(f"/experiment/{link}/config")
    except Session.DoesNotExist:
        return redirect(f"/experiment/{link}/config")

    # Look for the role of the participant and find the inputs that are listened
    agent = session.experiment.agents.get(role=request.session['role'])
    return render(request, "experiment/run.html", {'ws_setting': settings.WS_SETTING, "session": session, 'agent': agent, 'experiment': session.experiment})

@staff_member_required
def download_policy_template(request):
    """Serve the policy template file for download."""
    try:
        # Path to the policy template file
        template_path = os.path.join(settings.BASE_DIR, 'experiment', 'policy_template.py')

        # Read the template file
        with open(template_path, 'r') as f:
            content = f.read()

        # Create the HttpResponse with the file content
        response = HttpResponse(content, content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="policy_template.py"'

        return response
    except FileNotFoundError:
        raise Http404("Policy template file not found")


@staff_member_required
def download_environment_template(request):
    """Serve the environment template file for download."""
    try:
        # Path to the environment template file
        template_path = os.path.join(settings.BASE_DIR, 'experiment', 'environment_template.py')

        # Read the template file
        with open(template_path, 'r') as f:
            content = f.read()

        # Create the HttpResponse with the file content
        response = HttpResponse(content, content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="environment_template.py"'

        return response
    except FileNotFoundError:
        raise Http404("Environment template file not found")