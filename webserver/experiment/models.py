from django.db.models.functions import Now
from django.db import models
from django.core.exceptions import ValidationError

from accounts.models import Participant

import os




class Policy(models.Model):
    """
    The instance of an RL policy, i.e. a particular stochastic or deterministic mapping of states to actions.
    """
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)

    filepaths = models.JSONField('List of files needed by the policy', default=lambda: dict(policy='policy.py'), help_text='Those should be located under the top runner folder. You can also specify files under subfolders like <folder>/<filename>.py. <a href="/experiment/download/policy_template" target="_blank">Download policy template</a>, your policy file should define a variable called "policy" that implements the logic of the policy.')
    checkpoint_interval = models.IntegerField(default=0, help_text='Checkpoint interval (in steps) at which the policy is saved. Can also be 0 (never) or -2 (at the end of each episode).')
    
    metadata = models.JSONField(null=True, blank=True)

    def clean(self):
        """
        Validate the policy model:
        1. All files in filepaths can be found in the system
        2. There is one file stored in the filepaths dict with key "policy"
        3. That script defines a policy variable 
        """

        # Check that there is a 'policy' key in filepaths
        if 'policy' not in self.filepaths:
            raise ValidationError({
                'filepaths': 'Missing required "policy" key in filepaths dictionary.'
            })

        # Check that all files exist
        runner_path = os.path.join('..', 'runner')
        for key, filepath in self.filepaths.items():
            full_path = os.path.join(runner_path, filepath)
            if not os.path.exists(full_path):
                raise ValidationError({
                    'filepaths': f'File not found: {full_path}'
                })

        # Check that the policy script defines a policy variable
        policy_file_path = os.path.join(runner_path, self.filepaths['policy'])
        try:
            with open(policy_file_path, 'r') as file:
                file_content = file.read()
            # Simply search for a variable called "policy"
            if "policy =" not in file_content and "policy=" not in file_content:
                raise ValidationError({
                    'filepaths': f'Policy variable not defined in {policy_file_path}'
                })
        except Exception as e:
            raise ValidationError({
                'filepaths': f'Error parsing policy file {policy_file_path}: {str(e)}'
            })

class Agent(models.Model):
    """
    The union of policies and participants, i.e. an entity that perceives information from its environment, makes decisions based on those perceptions and internal goals, and acts to influence the environment’s future state.
    """
    role = models.CharField('Role in the environment (will be used as an ID)', max_length=255)
    name = models.CharField('Name that will be displayed', max_length=255)
    description = models.TextField(null=True, blank=True)

    policy = models.ForeignKey(Policy, on_delete=models.DO_NOTHING, null=True, blank=True)
    participant = models.BooleanField('Can the participant act?', default=False)

    keyboard_inputs = models.JSONField('Inputs captured from the participant, with mapping', default=lambda: dict(default=0))
    multiple_keyboard_inputs = models.BooleanField('Allow multiple inputs from users', default=False)
    inputs_type = models.CharField('How will the inputs be used in the environment?', choices=[('actions', 'To determine the agent\'s actions'), ('reward', 'To be used as reward'), ('other', 'To be given to the agent policy in a different way')], default='actions')
    textual_inputs = models.BooleanField('Allow textual inputs from users', default=False)
    
    metadata = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.name

    def clean(self):
        # Validate keyboard_inputs: must be a dict of strings
        if not isinstance(self.keyboard_inputs, dict):
            raise ValidationError({'keyboard_inputs': 'Keyboard inputs must be a dict.'})
        
class Environment(models.Model):
    """
    A simulated Markov decision process (MDP) adhering to the Gymnasium API with a tensor actions and rewards to support multi-agent and multi-objective settings.
    """
    name = models.CharField(max_length=255, null=False)
    description = models.TextField(null=True, blank=True)

    filepaths = models.JSONField('List of files needed by the environment', default=lambda: dict(environment='environment.py'), help_text='Those should be located under the top runner folder. You can also specify files under subfolders like <folder>/<filename>.py. <a href="/experiment/download/environment_template" target="_blank">Download environment template</a>, your environment file should define a variable called "environment" that implements the logic of the environment.')

    metadata = models.JSONField(null=True, blank=True)

    def clean(self):
        """
        Validate the environment model:
        1. All files in filepaths can be found in the system
        2. There is one file stored in the filepaths dict with key "environment"
        3. That script defines an environment variable
        """

        # Check that there is an 'environment' key in filepaths
        if 'environment' not in self.filepaths:
            raise ValidationError({
                'filepaths': 'Missing required "environment" key in filepaths dictionary.'
            })

        # Check that all files exist
        runner_path = os.path.join('..', 'runner')
        for key, filepath in self.filepaths.items():
            full_path = os.path.join(runner_path, filepath)
            if not os.path.exists(full_path):
                raise ValidationError({
                    'filepaths': f'File not found: {full_path}'
                })

        # Check that the environment script defines a variable called "environment"
        env_file_path = os.path.join(runner_path, self.filepaths['environment'])
        try:
            with open(env_file_path, 'r') as file:
                file_content = file.read()
            # Simply search for a variable called "environment"
            if "environment =" not in file_content and "environment=" not in file_content:
                raise ValidationError({
                    'filepaths': f'Environment variable not defined in {env_file_path}'
                })
        except Exception as e:
            raise ValidationError({
                'filepaths': f'Error parsing environment file {env_file_path}: {str(e)}'
            })
    def __str__(self):
        return self.name







class Experiment(models.Model):
    """
    A configured study combining zero, one or more RL agent(s), a single environment, one or more participants.
    """
    name = models.CharField('Name', max_length=100)
    link = models.CharField('Link to the experiment', max_length=100, unique=True, help_text="Unique identifier (slug) for the experiment URL")
    conda_environment = models.CharField('Name of a Conda environment already installed on the runner', max_length=100, null=True, blank=True)

    short_description = models.TextField('To be displayed on the home page')
    long_description = models.TextField('To be displayed on the experiment page')
    enabled = models.BooleanField(default=True)
    redirect_url = models.URLField(null=True, blank=True, help_text='If provided, participants will be redirected to this URL after completing the experiment.')

    environment = models.ForeignKey(Environment, on_delete=models.DO_NOTHING)
    agents = models.ManyToManyField(Agent, related_name='experiment', blank=True)
    number_of_episodes = models.IntegerField('Number of episodes to complete', default=1)

    target_fps = models.FloatField('Target FPS for the experiment', default=24.0, help_text='If you enable wait for inputs, the FPS will be determined by the participants\' inputs and this setting will be used as a forced pause after receiving inputs.')
    wait_for_inputs = models.BooleanField('Wait for participants\' input before moving to the next step', default=False)
    
    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name