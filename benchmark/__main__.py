#!/usr/bin/env python
"""
Run benchmark CLI when module is executed as script.

NOTE: Run from the webserver directory:
    cd webserver
    python ../benchmark/cli.py --help
"""

from .cli import main

if __name__ == "__main__":
    main()