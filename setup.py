#!/usr/bin/env python
from setuptools import setup, find_packages

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

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