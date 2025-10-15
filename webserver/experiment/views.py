from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import Http404
from experiment.forms import ConfigForm
from experiment import models

from mysite.settings import WS_SETTING


# Configuration view that will automatically check and save the parameters into the user session variable
@login_required
def config_(request, link):
    try:
        experiment = models.Experiment.objects.get(link=link)
    except models.Experiment.DoesNotExist:
        raise Http404("Experiment does not exist")
    
    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = ConfigForm(request.POST)
        form.fields['agent'].choices = experiment.agent_list
        # Check whether it's valid
        if form.is_valid():
            # Process the data in form.cleaned_data if the field has been filled
            for k in form.fields.keys():
                if k in form.cleaned_data.keys() and k != 'doc_link':  # Don't save doc_link to session
                    request.session[k] = form.cleaned_data[k]
            # Redirect to the run page
            return redirect(f"/experiment/{link}")

    # Create empty config form with the configuration options from the database
    form = ConfigForm()
    form.fields['agent'].choices = experiment.agent_list

    # Check if all the required fields have been saved in the user session variable
    saved = all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was already saved by the user, we create a prefilled form
    if(saved):
        initial_data = {k: request.session.get(k, None) for k in form.fields.keys() if k != 'doc_link'}
        form = ConfigForm(initial=initial_data)
        form.fields['agent'].choices = experiment.agent_list

    return render(request, "experiment/config.html", {'form': form, 'saved': saved, 'experiment_name': experiment.name})




@login_required
def run_(request, link):
    # Create empty config form
    form = ConfigForm()
    # Check if all the fields have been saved in the session
    saved = all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was not saved, we redirect the user to the config page
    if not saved:
        return redirect(f"/experiment/{link}/config")

    experiment = models.Experiment.objects.get(link=link)
    room_name = request.session['room_name']
    return render(request, "experiment/run.html", {"room_name": room_name, 'ws_setting': WS_SETTING, 'inputsListened': experiment.input_list, 'experiment_name': experiment.name, 'experiment_type': experiment.type})