#!/usr/bin/env python
"""
Master installation script for SHARPIE Gallery use cases.

Usage:
    sharpie-install <use_case> --gallery-dir <path>   # Install specific use case
    sharpie-install --all --gallery-dir <path>          # Install all use cases
    sharpie-install --list --gallery-dir <path>         # List available use cases
    sharpie-install <use_case> --check --gallery-dir <path>  # Validate without installing
    sharpie-install --quiet                             # Minimal output (errors only)
    sharpie-install --verbose                            # Detailed output
"""

import argparse
import sys
from pathlib import Path

from sharpie.install.installer import (
    install_use_case,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Install SHARPIE Gallery use cases')
    parser.add_argument('use_case', nargs='?', help='Use case to install')
    parser.add_argument('--all', action='store_true', help='Install all use cases')
    parser.add_argument('--list', action='store_true', help='List available use cases')
    parser.add_argument('--check', action='store_true', help='Validate without installing')
    parser.add_argument('--gallery-dir', type=str, default='../SHARPIE_Gallery',
                        help='Path to SHARPIE_Gallery directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (errors only)')
    parser.add_argument('--verbose', action='store_true', help='Detailed output')
    return parser.parse_args()


def get_verbosity(args):
    if args.quiet:
        return 0
    if args.verbose:
        return 2
    return 1


def list_use_cases(gallery_dir: Path):
    use_cases = []
    for item in gallery_dir.iterdir():
        if item.is_dir() and (item / 'config.yaml').exists():
            use_cases.append(item.name)
    return sorted(use_cases)


def main():
    args = parse_args()
    verbosity = get_verbosity(args)

    gallery_dir = Path(args.gallery_dir).resolve()
    if not gallery_dir.exists():
        print(f"Error: Gallery directory not found: {gallery_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print("Available use cases:")
        for uc in list_use_cases(gallery_dir):
            print(f"  - {uc}")
        return

    if args.all:
        failed = []
        for uc in list_use_cases(gallery_dir):
            try:
                install_use_case(uc, gallery_dir, check_only=args.check, verbosity=verbosity)
            except Exception as e:
                print(f"[FAIL] {uc} failed: {e}")
                failed.append(uc)
                continue

        if failed:
            print(f"\n[FAIL] {len(failed)} use case(s) failed")
            sys.exit(1)
    elif args.use_case:
        install_use_case(args.use_case, gallery_dir, check_only=args.check, verbosity=verbosity)
    else:
        argparse.ArgumentParser(description='Install SHARPIE Gallery use cases').print_help()


if __name__ == '__main__':
    main()