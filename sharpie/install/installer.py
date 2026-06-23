import os
import sys
import subprocess
from pathlib import Path

import yaml

from sharpie.install.validator import validate_single
from sharpie.install.utils import log

_django_initialized = False


def load_config(use_case: str, gallery_dir: Path) -> dict:
    config_path = gallery_dir / use_case / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def check_dependencies(config: dict, verbosity: int = 1):
    deps = config.get('dependencies', [])
    if not deps:
        log("  No dependencies required", level=2, verbosity=verbosity)
        return

    for dep in deps:
        log(f"  Installing {dep}...", level=1, verbosity=verbosity)
        pip_kwargs = {}
        if verbosity < 2:
            pip_kwargs['stdout'] = subprocess.DEVNULL
        if verbosity < 1:
            pip_kwargs['stderr'] = subprocess.DEVNULL
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep], **pip_kwargs)



def relative_to_absolute_paths(config: dict, use_case_dir: Path) -> dict:
    for object_key, object_value in config.items():
        if isinstance(object_value, dict) and 'filepaths' in object_value:
            for key, value in object_value['filepaths'].items():
                # Assumes that the filepath in the YAML config is only the filename
                config[object_key]['filepaths'][key] = str(use_case_dir / value)
    return config


def setup_database(config: dict, verbosity: int = 1):
    global _django_initialized

    if not _django_initialized:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sharpie.webserver.server.settings')
        import django
        django.setup()
        _django_initialized = True

    from sharpie.webserver.experiment.models import Environment, Experiment, Policy, Agent

    env_data = config['environment'].copy()
    env, created = Environment.objects.update_or_create(
        name=env_data['name'],
        defaults=env_data
    )
    log(f"  {'Created' if created else 'Updated'} Environment: {env.name}", level=1, verbosity=verbosity)

    pol = None
    if 'policy' in config:
        pol_data = config['policy'].copy()
        pol, created = Policy.objects.update_or_create(
            name=pol_data['name'],
            defaults=pol_data
        )
        log(f"  {'Created' if created else 'Updated'} Policy: {pol.name}", level=1, verbosity=verbosity)

    created_agents = []
    for agent_config in config['agents']:
        agent_data = agent_config.copy()
        if agent_data.get('policy'):
            agent_data['policy'] = Policy.objects.get(name=agent_data['policy'])
        else:
            agent_data['policy'] = None

        existing_agents = Agent.objects.filter(
            role=agent_config['role'],
            name=agent_config['name']
        )

        if existing_agents.count() > 1:
            keep_agent = existing_agents.first()
            existing_agents.exclude(pk=keep_agent.pk).delete()
            for key, val in agent_data.items():
                setattr(keep_agent, key, val)
            keep_agent.save()
            agent, created = keep_agent, False
        elif existing_agents.count() == 1:
            agent = existing_agents.first()
            for key, val in agent_data.items():
                setattr(agent, key, val)
            agent.save()
            created = False
        else:
            agent = Agent.objects.create(**agent_data)
            created = True

        created_agents.append(agent)
        log(f"  {'Created' if created else 'Updated'} Agent: {agent.name} ({agent.role})", level=1, verbosity=verbosity)

    exp_data = config['experiment'].copy()
    exp_data['environment'] = env
    exp, created = Experiment.objects.update_or_create(
        link=exp_data['link'],
        defaults=exp_data
    )

    for agent in created_agents:
        exp.agents.add(agent)

    log(f"  {'Created' if created else 'Updated'} Experiment: {exp.name} (link: {exp.link})", level=1, verbosity=verbosity)


def show_installation_notes(config: dict, verbosity: int = 1):
    notes = config.get('installation_notes')
    if notes:
        log(f"\nInstallation Notes:", level=1, verbosity=verbosity)
        log(f"   {notes}\n", level=1, verbosity=verbosity)


def install_use_case(use_case: str, gallery_dir: Path, check_only: bool = False, verbosity: int = 1):
    action = 'Validating' if check_only else 'Installing'
    log(f"\n{action} {use_case}...", level=1, verbosity=verbosity)

    config = load_config(use_case, gallery_dir)

    show_installation_notes(config, verbosity)

    log("Step 1/4: Checking dependencies...", level=1, verbosity=verbosity)
    check_dependencies(config, verbosity)

    log("\nStep 2/4: Validating files...", level=1, verbosity=verbosity)
    validate_single(use_case, gallery_dir, verbosity)

    if check_only:
        return

    use_case_dir = gallery_dir / use_case

    log("\nStep 3/4: Converting relative filepaths to absolute filepaths...", level=1, verbosity=verbosity)
    config = relative_to_absolute_paths(config, use_case_dir)

    log("\nStep 4/4: Setting up database...", level=1, verbosity=verbosity)
    setup_database(config, verbosity)

    log(f"\n[OK] {use_case} installed successfully!", level=1, verbosity=verbosity)