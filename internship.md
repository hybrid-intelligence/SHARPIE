this is the readme file that is intended to guide new interns contributing to SHARPIE.

## About the author
I am Omer E. Aras, I was a summer intern under the supervision of dr. Kim Baraka in summer 2025. I have extended the current version (as of June 2025) with 3 different applications: Panda, Custom Maze and Tamer Maze.

I presume you have successfully done the required installations stated in the README.md file of the SHARPIE, and ready to work.If not so, please follow the instructions there, ask the developers for help. Here I will give a brief information about the structure of the project and what you need to do to be able to write your own applications without facing an issue.

## Structure
There are a variety of folders within SHARPIE project, each corresponding to a different use-case of how human input can be integrated into RL training scheme (e.g. evaluative feedback, demonstration).

Common backend logic that each application inherits is implemented in 'sharpie' folder. The remaining folders stand for a different application (e.g. AMaze,, Mountain Car)

Each of the application includes more or less similar types of files. 

- Most of the application folders do have agent.py which defines the model -if exists- and how will that model be trained. This file is used to define & determine the RL algorithm to be used. 

- Websocket.py is the core file that runs the application of interest. It behaves like the environment in the context of RL, includes a step function that is identical with RL steps. After every step call, we call 'output' function (misspelled as 'ouputs' by the developers) which renders the new image, new state of the environment and the agent.

- forms.py defines the configure attributes that is presented to the user for the application setting.

there are other files too but you dont need to care about them unless you are changing the main logic of the backend, which I believe is unlikely for a summer intern, Libio is and probably will be in charge of core changes to the repository. Now there is the last-but-not-least matter:

When creating your application, you need to add your application(s)'s name throughout the repository. 
- sharpie/asgi.py -> add your imports and re_paths just like in the existing code.
- sharpie/urls.py -> add url pattern of your application like in the code.
- sharpie/settings.py -> write your app name into 'INSTALLED_APPS' accordingly.

Also adjust your application based on the existing modules / applications. Please feel free to copy paste for initial configuration, in fact I would urge you to do so, as not copying the existing code would likely end up with errors of incompatibility with the SHARPIE logic.


## P.S.
I genuinely had the best time working with my supervisor dr. Kim, current SHARPIE developers Floris, Libio, Yeqian, Kevin and Social AI lab members Lin, Sal, Bernhard, Muhan, and so on. I could not even had a chance to exchange proper farewells with most of them :/  I am deeply honored and humbled with all the kindness and openness of all the people there. I am genuinely grateful that I got this opportunity and would definitely recommend to take part in this project to anyone who has a passion for RL and likes to work with websites :) 


You can always reach me out via my email account:
omerediparas@gmail.com


I wish you a good time with your project assignments and in life.