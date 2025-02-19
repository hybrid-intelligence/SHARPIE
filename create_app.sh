#! /bin/sh

if [ "$1" == "-h" ]; then
    echo "Usage: `basename $0` app_name app_folder"
    exit 0
fi
if [ "$#" -ne 2 ]; then
    echo "You are supposed to enter 2 parameters. Type `basename $0` -h for more information."
    exit 0
fi

# We get the app and folder name
app_name=$1
app_folder=$2
# We copy the example application
cp -r example/ $app_folder
# We set the variables in the settings
echo "app_name = \"$app_name\"
app_folder = \"$app_folder\"" > $app_folder/settings.py
# We change the name of the static and templates folder accordingly
mv $app_folder/static/example $app_folder/static/$app_folder
mv $app_folder/templates/example $app_folder/templates/$app_folder
# The app is ready, we display the instructions to integrate it
echo 'The app is ready! To make it available on your website:'
echo '\t add the folder name to INSTALLED_APPS in mysite/settings.py'
echo '\t add the folder name to the urlpatterns in mysite/urls.py'
echo '\t add the folder name to the websocket in mysite/asgi.py'
echo '\t in your new app, (at least) modify static/js/script.js and websocket.py to suit your needs!'