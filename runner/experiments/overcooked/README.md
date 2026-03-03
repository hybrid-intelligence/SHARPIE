# Installation
Do not run `pip install overcooked-ai`.

Do use [git sources](https://github.com/HumanCompatibleAI/overcooked_ai?tab=readme-ov-file#building-from-source-)

Requires python 3.10

# Database setup

Run from the webserver directory (e.g. `cd webserver`):

```bash
python manage.py shell -c "
from experiment.models import Environment;

# Add environment to the database 
Environment.objects.update_or_create(
  name='Overcooked',
  defaults={
    'description':'Serve as many orders as possible with an AI teammate',
    'filepaths':{
      'environment':'experiments/overcooked/environment.py'
    }
  }
)

# Add this experiment to the database 
from experiment.models import Experiment, Environment;
 
Experiment.objects.update_or_create(
  link='overcooked', defaults={
    'name': 'Overcooked',
    'short_description': 'Human-AI collaborative cooking',
    'long_description': 'Serve as many orders as possible with an AI teammate',
    'enabled': True,
    'environment': Environment.objects.get(name='Overcooked'),
    'number_of_episodes': 1,
    'target_fps': 4.0,
    'wait_for_inputs': True
  }
)

# Add this policy to the database 
from experiment.models import Policy
Policy.objects.update_or_create(
  name='OvercookedRandom',
  defaults={
    'description': 'Random agent for overcooked',
    'filepaths': {'policy': 'experiments/overcooked/policy.py'},
    'checkpoint_interval': 0
  }
)

# Add this agent to the database 
from experiment.models import Agent, Policy;
Agent.objects.update_or_create(
  role='Agent',
  defaults={
    'name': 'Agent',
    'description': 'Artificial player',
    'policy': Policy.objects.get(name='OvercookedRandom'),
    'participant': False,
    'keyboard_inputs': {'ArrowUp': 1, 'ArrowDown': -1, 'default': 0},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
  }
)
Agent.objects.update_or_create(
  role='Human',
  defaults={
    'name': 'Human',
    'description': 'Human player',
    'policy': None,
    'participant': True,
    'keyboard_inputs': {'ArrowUp': 0, 'ArrowDown': 1, 'ArrowRight': 2, 'ArrowLeft': 3, 'Enter': 5, 'default': 4},
    'keyboard_input_display': {
      'ArrowUp': {'symbol': '↑', 'label': 'Up'},
      'ArrowDown': {'symbol': '↓', 'label': 'Down'},
      'ArrowRight': {'symbol': '→', 'label': 'Right'},
      'ArrowLeft': {'symbol': '←', 'label': 'Left'},
      'Enter': {'symbol': '↵', 'label': 'Enter'}
    },
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': False
  }
)

# Link agent to experiment 
from experiment.models import Experiment, Agent;
exp = Experiment.objects.get(link='overcooked');
exp.agents.add(Agent.objects.get(role='Agent'));
exp.agents.add(Agent.objects.get(role='Human'))

# Register runner 
from runner.models import Runner;
Runner.objects.update_or_create(connection_key='i-am-secret')"
```