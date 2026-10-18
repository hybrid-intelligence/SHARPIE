"""
Export an Experiment from the database to a SHARPIE Gallery ``config.yaml``.

This is the inverse of :func:`sharpie.install.installer.setup_database`: the
produced YAML can be installed again with ``sharpie-install``, provided the
code files listed under ``filepaths`` are placed next to the ``config.yaml``.
"""

import os

import yaml


class ExportError(ValueError):
    """Raised when an experiment cannot be represented in the config.yaml format."""


def _export_filepaths(filepaths: dict) -> dict:
    """Reduce stored filepaths to bare filenames, as expected next to config.yaml."""
    return {key: os.path.basename(path) for key, path in filepaths.items()}


def _check_unique_filenames(*filepaths_dicts: dict) -> None:
    """Refuse files that would share a filename once placed next to config.yaml."""
    filenames = [os.path.basename(path)
                 for filepaths in filepaths_dicts for path in filepaths.values()]
    # All files go in one folder next to config.yaml, so every filename must be unique
    if len(set(filenames)) != len(filenames):
        duplicates = sorted({name for name in filenames if filenames.count(name) > 1})
        raise ExportError(f"Several files share the same filename: {', '.join(duplicates)}")


def _export_instance(instance, exclude=()):
    """
    Return the configurable fields of a model instance as a dict, in model order.
    """
    data = {}
    for field in instance._meta.concrete_fields:
        # Skip the id, automatic fields (not editable), and fields that are excluded
        if field.primary_key or not field.editable or field.name in exclude:
            continue
        value = getattr(instance, field.name)
        if field.is_relation:
            value = value.name if value is not None else None
        data[field.name] = value
    if 'filepaths' in data:
        data['filepaths'] = _export_filepaths(data['filepaths'])
    return data


def export_experiment(experiment):
    """
    Build the config.yaml content of an experiment as a dictionary.

    Raises ExportError if the experiment cannot be represented in config.yaml.
    """
    # Fixed order so repeated exports are identical
    agents = list(experiment.agents.select_related('policy').order_by('id'))

    # Key by pk, so two different policies with the same name still count as two
    policies = {agent.policy.pk: agent.policy for agent in agents if agent.policy}

    # Only one policy per experiment is supported by config.yaml
    if len(policies) > 1:
        names = ', '.join(sorted(policy.name for policy in policies.values()))
        raise ExportError(f"Experiment '{experiment.link}' uses several policies ({names}); "
                          "config.yaml supports only one.")

    policy_filepaths = [policy.filepaths for policy in policies.values()]
    _check_unique_filenames(experiment.environment.filepaths, *policy_filepaths)

    config = {
        'use_case': experiment.link,
        'dependencies': experiment.dependencies,
        'environment': _export_instance(experiment.environment),
    }

    if policies:
        config['policy'] = _export_instance(next(iter(policies.values())))

    config['agents'] = [_export_instance(agent) for agent in agents]
    # The environment and dependencies are already written as top-level blocks
    config['experiment'] = _export_instance(experiment, exclude=('environment', 'dependencies'))

    return config


class _ConfigDumper(yaml.SafeDumper):
    """YAML dumper writing multi-line strings as readable ``|`` blocks."""


def _represent_str(dumper, value):
    """Use the block style for multi-line strings, the default style otherwise."""
    style = '|' if '\n' in value else None
    return dumper.represent_scalar('tag:yaml.org,2002:str', value, style=style)


_ConfigDumper.add_representer(str, _represent_str)


def export_experiment_yaml(experiment):
    """
    Export an experiment as config.yaml text.
    """
    return yaml.dump(export_experiment(experiment), Dumper=_ConfigDumper,
                     sort_keys=False, allow_unicode=False)
