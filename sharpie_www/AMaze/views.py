from django.shortcuts import render, redirect

import os

from .forms import ConfigForm, RunForm




def config_(request):
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        form = ConfigForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            for k in form.fields.keys():
                # If the field has been filled
                if k in form.cleaned_data.keys():
                    request.session[k] = form.cleaned_data[k]

    # Create empty form
    form = ConfigForm()
    # Check if all the fields have been saved in the session
    saved =  all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was already saved by the user, we create a prefilled form
    if(saved):
        form = ConfigForm(initial={k:request.session.get(k, None) for k in form.fields.keys()})

    app_name  = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    return render(request, app_name+"/config.html", {"form": form, 'saved': saved})





def run_(request):
    app_name  = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    
    # Create empty form
    form = ConfigForm()
    # Check if all the fields have been saved in the session
    saved =  all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was already saved by the user, we create a prefilled form
    if not saved:
        return redirect("/"+app_name+"/config")

    form = RunForm()
    room_name = request.session['room_name']
    return render(request, app_name+"/run.html", {"room_name": room_name, "app_name": app_name, "form": form})