#! /usr/bin/env python
# pylint: disable=invalid-name

"""Script that runs all verification steps.
"""

import argparse
import sys
from subprocess import CalledProcessError, run


def main(arguments):
    try:
        print("Verifying with " + str(arguments))
        if "pytest" in args:
            print("Tests (pytest):", flush=True)
            run("pytest -v --cov=vampire --color=yes vampire", shell=True, check=True)

        if "pylint" in arguments:
            print("Linter (pylint):", flush=True)
            run("pylint -d locally-disabled,locally-enabled -f colorized vampire", shell=True, check=True)
            print("pylint checks passed")

        if "mypy" in arguments:
            print("Typechecker (mypy):", flush=True)
            run("mypy vampire --ignore-missing-imports", shell=True, check=True)
            print("mypy checks passed")

        if "check-large-files" in arguments:
            print("Checking all added files have size <= 5MB", flush=True)
            run("./scripts/check_large_files.sh 5", shell=True, check=True)
            print("check large files passed")

    except CalledProcessError:
        # squelch the exception stacktrace
        sys.exit(1)

if __name__ == "__main__":

    checks = ['pytest', 'pylint', 'mypy', 'check-large-files']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', type=str, required=False, nargs='+', choices=checks)

    args = parser.parse_args()

    if args.checks:
        run_checks = args.checks
    else:
        run_checks = checks

    main(run_checks)
