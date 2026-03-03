# Installation
Run `pip install 'pettingzoo[mpe]'`

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Simple spread', defaults={'description':'Simple experiment where 3 agents','filepaths':{'environment':'spread/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='spread', defaults={
    'name': 'Simple spread',
    'short_description': 'Simple spread experiment',
    'long_description': 'Simple spread experiment. Each agent should cover one target.',
    'enabled': True,
    'environment': Environment.objects.get(name='Simple spread'),
    'number_of_episodes': 1,
    'target_fps': 24.0,
    'wait_for_inputs': False
})"
```

<!-- Add this policy to the database -->
```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='Spread agent', defaults={
    'description': '',
    'filepaths': {'policy': 'spread/policy.py'},
    'checkpoint_interval': 0
})"
```

<!-- Add these agents to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_0', defaults={
    'name': 'Agent 1',
    'description': '',
    'policy': Policy.objects.get(name='Spread agent'),
    'participant': True,
    'keyboard_inputs': {'ArrowLeft': 1, 'ArrowRight': 2, 'ArrowDown': 3, 'ArrowUp': 4, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
})"
```

```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_1', defaults={
    'name': 'Agent 2',
    'description': '',
    'policy': Policy.objects.get(name='Spread agent'),
    'participant': False,
    'keyboard_inputs': {'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
})"
```

```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_2', defaults={
    'name': 'Agent 3',
    'description': '',
    'policy': Policy.objects.get(name='Spread agent'),
    'participant': False,
    'keyboard_inputs': {'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
})"
```

<!-- Link agents to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='spread'); exp.agents.set([Agent.objects.get(role='agent_0'), Agent.objects.get(role='agent_1'), Agent.objects.get(role='agent_2')])"
```