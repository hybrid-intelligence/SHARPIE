#!/usr/bin/env python
"""Generate DEPENDENCIES_PIP.md with license information for direct dependencies."""

import re
import subprocess
from pathlib import Path


def get_packages_from_requirements(filepath):
    """Extract package names from a requirements file."""
    packages = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg = re.split(r'[\[=<>!~]', line)[0].strip()
                if pkg:
                    packages.append(pkg)
    return packages


def main():
    root = Path(__file__).parent.parent
    
    packages = []
    packages.extend(get_packages_from_requirements(root / 'requirements.txt'))
    packages.extend(get_packages_from_requirements(root / 'docs' / 'requirements.txt'))
    
    result = subprocess.run(
        ['pip-licenses', '--from=classifier', '--format=markdown', 
         '-p'] + packages + ['--output-file=DEPENDENCIES_PIP.md'],
        capture_output=True,
        text=True,
        cwd=root
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return 1
    
    output_file = root / 'DEPENDENCIES_PIP.md'
    content = output_file.read_text()
    
    header = """# Python Dependencies License Overview

This file lists all Python package dependencies and their licenses.

To regenerate this file after changing requirements, run:

    python scripts/generate_dependencies.py

"""

    output_file.write_text(header + content)
    print(f"Generated {output_file}")
    return 0


if __name__ == '__main__':
    exit(main())