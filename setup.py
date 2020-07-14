#!/usr/bin/env python
from setuptools import setup, find_packages

from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='none')

# reqs is a list of requirement
dependencies = [str(ir.req) for ir in install_reqs]

VERSION = '0.1.0'

setup(name='vampire',
      version=VERSION,
      author='Suchin Gururangan',
      #install_requires=dependencies,
      # Package info
      packages=find_packages(exclude=('test',)),
      zip_safe=True)