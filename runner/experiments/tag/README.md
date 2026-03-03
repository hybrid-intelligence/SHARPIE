# Installation
Run `pip install 'pettingzoo[mpe]'`

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Simple tag', defaults={'description':'','filepaths':{'environment':'tag/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='tag', defaults={
    'name': 'Simple tag',
    'short_description': 'Simple tag experiment',
    'long_description': 'Simple tag experiment, this requires 2 simultaneous users. The adversary needs to chase the agent.',
    'enabled': True,
    'environment': Environment.objects.get(name='Simple tag'),
    'number_of_episodes': 1,
    'target_fps': 24.0,
    'wait_for_inputs': False
})"
```

<!-- Add these agents to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent; Agent.objects.update_or_create(role='agent_0', defaults={
    'name': 'Fleeing agent',
    'description': 'Fleeing agent',
    'policy_id': None,
    'participant': True,
    'keyboard_inputs': {'ArrowLeft': 1, 'ArrowRight': 2, 'ArrowDown': 3, 'ArrowUp': 4, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
})"
```

```bash
python manage.py shell -c "from experiment.models import Agent; Agent.objects.update_or_create(role='adversary_0', defaults={
    'name': 'Adversary agent',
    'description': '',
    'policy_id': None,
    'participant': True,
    'keyboard_inputs': {'ArrowLeft': 1, 'ArrowRight': 2, 'ArrowDown': 3, 'ArrowUp': 4, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
})"
```

<!-- Link agents to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='tag'); exp.agents.set([Agent.objects.get(role='agent_0'), Agent.objects.get(role='adversary_0')])"
```