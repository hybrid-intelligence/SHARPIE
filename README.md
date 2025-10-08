[![versions](https://img.shields.io/badge/python-3.10-blue)](#) [![motivating-paper](https://img.shields.io/badge/paper-motivation-blue)](https://doi.org/10.48550/arXiv.2501.19245)

# SHARPIE - beta version
## Shared Human-AI Reinforcement Learning Platform for Interactive Experiments
[![Demo](https://github.com/libgoncalv/SHARPIE/blob/main/home/static/home/preview_image_1.png)](https://archive.org/embed/hhai-demo-1)

Our framework is relying on Django for serving files to the user browser. Therefore, it is following the architecture of Django to organize the code i.e. decomposition of the pages in apps. In this repository, we only present the core of SHARPIE and you can find examples/use-cases in our Gallery repository. For clarity we will detail briefly here the important files in our project but we highly recommend to look at the Django documentation for a better understanding.


## Development installation
* We highly recommend to use a virtual environment such as Anaconda. This code has been tested on Python 3.11.
* Git clone this repository
* Install Redis server `apt install redis-server`
* Install SHARPIE requirements `pip install -r requirements`
* Webserver:
  * Run `cd webserver`
  * Run `python manage.py runserver` to start the webserver
* You can access the website at [localhost:8000](localhost:8000) and manage the authorized users from [localhost:8000/admin](localhost:8000/admin) with the username "admin" and password "password"
* For now there is no experiment available but you can find some examples ready to use in our gallery!

## Run into production
You can start by looking at the [deployement checklist](https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/) from Django. For the webserver, we recommend to use the [example setup with Nginx and Supervisor](https://channels.readthedocs.io/en/latest/deploying.html#example-setups) from the official Channels documentation. For the runner, we also recommend using Supervisor:
* Copy `runner_supervisor.conf` to `/etc/supervisor/conf.d/` and modify the paths mentionned in the file to match your configuration
* Have supervisor reread and update its jobs: `sudo supervisorctl reread && sudo supervisorctl update`

## FAQ
**Q**: *How long will SHARPIE be supported?*  
A: SHARPIE is currently under active development, a request for support budget (a.o.) for 2025
is in preparation and we will continue do so until the end of the [HI
center](https://www.hybrid-intelligence-centre.nl/) in 2029.

**Q**: *Can SHARPIE integrate with environmnet X?*  
A: SHARPIE integrates with any environment that implements the
[Gymnasium ``env``](https://gymnasium.farama.org/api/env/) API: SHARPIE needs access to 
``step()``, ``reset()`` and ``render()`` functions. Since SHARPIE runs the environment on a
back-end server rather than in the browser, it supports any environment with Python bindings to these
functions.

**Q**: *Does SHARPIE support experiments involving mixed human-AI teams?*  
A: SHARPIE is designed to support mixed human-AI teams, involving multiple AI agents and multiple
human participants at the same time. The SHARPIE architecture includes a back-end to manage
synchronization between participants, and rendering for you.

**Q**: *What are the computational requirements for SHARPIE*  
A: SHARPIE itself does not come with strict computational requirements, we suggest that you follow
the computational requirements of your environment and RL model of choice.

**Q**: *Can I use SHARPIE to run experiments involving participants from different continents?*  
A: It depends. Having participants located in different continents in the same room is likely to
cause latency issues that are fundamental to the nature of cross-continental networking.
Separating participants per room may be a viable alternative if your experimental setup support
this.

**Q**: *Why did you develop SHARPIE?*  
A: We developed SHARPIE to facilitate experiments involving RL agents and humans. We believe that
the study of this interaction is crucial to establishing artificial intelligence(s) that do not
replace human intellect but instead expand it. We thereby want to put humans at the centre,
and change the course of the ongoing AI revolution.

**Q**: *Can SHARPIE be used for education and outreach?*  
A: SHARPIE can be used for educational purposes and outreach. However, it is currently still under
active development. We plan to develop educational materials on hybrid human-AI systems and
experiments once the platform stabilizes.

## Acknowledgements
This research was funded by the [Hybrid Intelligence
Center](https://hybridintelligence-centre.nl), a 10-year programme funded by the Dutch Ministry of
Education, Culture and Science through the Netherlands Organisation for Scientific Research, Grant
No: 024.004.022.

## Citations
When using this project in a scientific publication please cite:
```bibtex
@inproceedings{sharpiecaihu25,
    booktitle = {AAAI Bridge Program Workshop on Collaborative AI and Modeling of Humans},
    title = {{SHARPIE}: A Modular Framework for Reinforcement Learning and Human-AI Interaction Experiments},
    author = {Ayd$\i$n, H{\"{u}}seyin and Godin-Dubois, Kevin and Goncalves Braz, Libio and den Hengst,
    Floris and Baraka, Kim and {\c{C}}elikok, Mustafa Mert and Sauter, Andreas and Wang, Shihan and
    Oliehoek, Frans A},
    month = {feb},
    address = {Philadelphia, Pennsylvania, USA},
    doi={10.48550/arXiv.2501.19245},
    year = {2025}
}
```
