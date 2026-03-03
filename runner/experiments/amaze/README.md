# Installation
Run `pip install 'amaze-benchmarker[all]'`

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Amaze', defaults={'description':'A lightweight maze navigation task generator for sighted AI agents','filepaths':{'environment':'amaze/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='amaze', defaults={
    'name': 'Amaze',
    'short_description': 'A lightweight maze navigation task',
    'long_description': 'A lightweight maze navigation task generator for sighted AI agents.',
    'enabled': True,
    'environment': Environment.objects.get(name='Amaze'),
    'number_of_episodes': 1,
    'target_fps': 1.0,
    'wait_for_inputs': True
})"
```

<!-- Add this policy to the database -->
```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='Amaze', defaults={
    'description': 'TAMER policy for Amaze',
    'filepaths': {'policy': 'amaze/policy.py', 'tamer': 'amaze/tamer.py'},
    'checkpoint_interval': 1
})"
```

<!-- Add this agent to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_0', defaults={
    'name': 'Agent',
    'description': '',
    'policy': Policy.objects.get(name='Amaze'),
    'participant': True,
    'keyboard_inputs': {'ArrowUp': 1, 'ArrowDown': -1, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'reward',
    'textual_inputs': False
})"
```

<!-- Link agent to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='amaze'); exp.agents.add(Agent.objects.get(role='agent_0'))"
```