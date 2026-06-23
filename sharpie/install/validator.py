import sys
import importlib.util
import inspect
from pathlib import Path

import yaml

from sharpie.install.utils import log


def validate_single(use_case: str, gallery_dir: Path, verbosity: int = 1):
    use_case_dir = gallery_dir / use_case
    config_path = use_case_dir / 'config.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, str(use_case_dir))
    try:
        env_config = config['environment']
        env_file = env_config['filepaths']['environment'].split('/')[-1]
        env_path = use_case_dir / env_file

        if not env_path.exists():
            raise FileNotFoundError(f"Missing environment: {env_path}")

        env_module_name = f"{use_case}_environment"
        spec = importlib.util.spec_from_file_location(env_module_name, env_path)
        env_mod = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(env_mod)
        except Exception as e:
            raise RuntimeError(f"Failed to load {env_path}: {e}")

        if not hasattr(env_mod, 'environment'):
            raise ValueError(f"{env_path} must define 'environment' variable")

        env = env_mod.environment
        required_methods = ['reset', 'step', 'render']
        for method in required_methods:
            if not hasattr(env, method):
                raise ValueError(f"environment must have {method}() method")

        log(f"  [OK] Environment validated: {env_config['name']}", level=1, verbosity=verbosity)

        pol_mod = None
        if 'policy' in config:
            pol_config = config['policy']

            for key, filepath in pol_config['filepaths'].items():
                pol_file = filepath.split('/')[-1]
                pol_path = use_case_dir / pol_file
                if not pol_path.exists():
                    raise FileNotFoundError(f"Missing {key}: {pol_path}")

            pol_file = pol_config['filepaths']['policy'].split('/')[-1]
            pol_path = use_case_dir / pol_file

            pol_module_name = f"{use_case}_policy"
            spec = importlib.util.spec_from_file_location(pol_module_name, pol_path)
            pol_mod = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(pol_mod)
            except Exception as e:
                raise RuntimeError(f"Failed to load {pol_path}: {e}")

            if not hasattr(pol_mod, 'policy'):
                raise ValueError(f"{pol_path} must define 'policy' variable")

            policy_class = pol_mod.policy.__class__.__name__
            if policy_class != 'Policy':
                raise ValueError(f"Policy class should be named 'Policy', got '{policy_class}'")

            sig = inspect.signature(pol_mod.policy.predict)
            params = list(sig.parameters.keys())
            if 'observation' not in params:
                raise ValueError("predict() must accept 'observation' parameter")

            log(f"  [OK] Policy validated: {pol_config['name']}", level=1, verbosity=verbosity)

        log("  [OK] Interface test passed", level=1, verbosity=verbosity)
    finally:
        sys.path.remove(str(use_case_dir))