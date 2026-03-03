# Installation
Run `pip install gymnasium[classic_control]`

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Mountain car TAMER', defaults={'description':'','filepaths':{'environment':'mountain_tamer/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='mountain_tamer', defaults={
    'name': 'Mountain car TAMER',
    'short_description': 'TAMER mountain care experiment by Knox & Stone',
    'long_description': 'This is a classic experiment by Knox & Stone in which an agent is trained with evaluative feedback.',
    'enabled': True,
    'environment': Environment.objects.get(name='Mountain car TAMER'),
    'number_of_episodes': 50,
    'target_fps': 6.666,
    'wait_for_inputs': False
})"
```

<!-- Add this policy to the database -->
```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='Mountain car TAMER', defaults={
    'description': '',
    'filepaths': {'policy': 'mountain_tamer/policy.py', 'tamer': 'mountain_tamer/tamer.py'},
    'checkpoint_interval': 1
})"
```

<!-- Add this runner to the database -->
```bash
python manage.py shell -c "from runner.models import Runner; Runner.objects.update_or_create(connection_key='mct-secretkey')"
```

<!-- Add this agent to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_mct', defaults={
    'name': 'Mountain car TAMER agent',
    'description': '',
    'policy': Policy.objects.get(name='Mountain car TAMER'),
    'participant': True,
    'keyboard_inputs': {'ArrowUp': 1, 'ArrowDown': -1, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'reward',
    'textual_inputs': False
})"
```


<!-- Link agent to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='mountain_tamer'); exp.agents.add(Agent.objects.get(role='agent_mct'))"
```

<!-- copy the run.HTML for a custom template -->
```bash
cp mountain_tamer/run_tamer.html ../SHARPIE/webserver/experiment/templates/experiment/run.html
```

```bash
cp -r mountain_tamer ../SHARPIE/runner/experiments/
```