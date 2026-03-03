# Installation
Run `pip install gymnasium`

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Mountain car', defaults={'description':'','filepaths':{'environment':'mountain/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='mountain', defaults={
    'name': 'Mountain car',
    'short_description': 'Simple mountain car experiment - Single player',
    'long_description': 'Whatever i put here, it will be given to the participant.\r\n\n<br>\n<br>\nBla blablabla blabla balablabla',
    'enabled': True,
    'environment': Environment.objects.get(name='Mountain car'),
    'number_of_episodes': 1,
    'target_fps': 24.0,
    'wait_for_inputs': False
})"
```

<!-- Add this agent to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent; Agent.objects.update_or_create(role='agent_1', defaults={
    'name': 'Car',
    'description': 'Mountain car',
    'policy_id': None,
    'participant': True,
    'keyboard_inputs': {'ArrowLeft': 0, 'ArrowRight': 2, 'default': 1},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
})"
```

<!-- Link agent to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='mountain'); exp.agents.add(Agent.objects.get(role='agent_1'))"
```