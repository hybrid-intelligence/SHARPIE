#! /bin/sh

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
echo '\d add the folder name to INSTALLED_APPS in mysite/settings.py'
echo '\d add the folder name to the urlpatterns in mysite/urls.py'
echo '\d add the folder name to the websocket in mysite/asgi.py'
echo '\d in your new app, (at least) modify static/js/script.js and websocket.py to suit your needs!'