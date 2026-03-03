# SHARPIE Gallery
## Shared Human-AI Reinforcement Learning Platform for Interactive Experiments Gallery

This repository presents use-cases for [SHARPIE](https://github.com/libgoncalv/SHARPIE/). 

### Usage instructions
* Install SHARPIE by following the instructions on the main repository
* Choose one (or several) of the use-cases and follow the instructions in the README.md file
* Copy the experiments you selected under the top folder called runner
* Lauch the webserver as explained on the SHARPIE repository, register your runner with a secret connection key and in a different terminal launch the runner using `cd runner && python manage.py runserver --connection-key secret`.

### Content of the use-cases
Each use-case folder contains configuration files for the backend environment and agent(s). This includes:
* environment.py, this can be any environment as long as it defines reset(), step(action) and render()
* policy.py, this can be any class as long as it defines predict(obs) and optionnally update(state, action, reward, done, next_state) if you want to perform live training
