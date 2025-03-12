from django.shortcuts import render, redirect
from .forms import ConfigForm

from .settings import app_name, app_folder
from sharpie.settings import WS_SETTING


# Configuration view that will automatically check and save the parameters into the user session variable
def config_(request):
    # If this is a POST request we need to process the form data
    if request.method == "POST":
        # Create a form instance and populate it with data from the request:
        form = ConfigForm(request.POST)
        # Check whether it's valid
        if form.is_valid():
            # Process the data in form.cleaned_data if the field has been filled
            for k in form.fields.keys():
                if k in form.cleaned_data.keys():
                    request.session[k] = form.cleaned_data[k]

    # Create empty config form
    form = ConfigForm()
    # Check if all the required fields have been saved in the user session variable
    saved =  all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was already saved by the user, we create a prefilled form
    if(saved):
        form = ConfigForm(initial={k:request.session.get(k, None) for k in form.fields.keys()})

    return render(request, app_folder+"/config.html", {'form': form, 'app_name': app_name, 'saved': saved})





def run_(request):
    # Create empty config form
    form = ConfigForm()
    # Check if all the fields have been saved in the session
    saved =  all((k in request.session.keys() or not form.fields[k].required) for k in form.fields.keys())
    # If a config was not saved, we redirect the user to the config page
    if not saved:
        return redirect("/"+app_folder+"/config")

    room_name = request.session['room_name']
    return render(request, app_folder+"/run.html", {"room_name": room_name, 'ws_setting': WS_SETTING, "app_name": app_name, "app_folder": app_folder})