# Installation
Run `pip install gymnasium`

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Frozen lake', defaults={'description':'','filepaths':{'environment':'frozen/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='frozen', defaults={
    'name': 'Frozen lake',
    'short_description': 'Frozen lake experiment',
    'long_description': 'Bla blabla blabla',
    'enabled': True,
    'environment': Environment.objects.get(name='Frozen lake'),
    'number_of_episodes': 3,
    'target_fps': 1.0,
    'wait_for_inputs': True
})"
```

<!-- Add this policy to the database -->
```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='Frozen', defaults={
    'description': '',
    'filepaths': {'policy': 'frozen/policy.py', 'tamer': 'frozen/tamer.py'},
    'checkpoint_interval': 1
})"
```

<!-- Add this agent to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent', defaults={
    'name': 'Frozen agent',
    'description': '',
    'policy': Policy.objects.get(name='Frozen'),
    'participant': True,
    'keyboard_inputs': {'ArrowUp': 1, 'ArrowDown': -1, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'reward',
    'textual_inputs': False
})"
```

<!-- Link agent to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='frozen'); exp.agents.add(Agent.objects.get(role='agent'))"
```

* Train `True`