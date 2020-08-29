from setuptools import setup
import setuptools
import os.path as op

# suggest to install by
# python setup.py build develop --user

req = '../requirements.txt'
if op.isfile(req):
    with open(req) as fp:
        requirements = fp.readlines()
else:
    requirements = []

setup(name='qd',
      version='0.1',
      description='Provide basic utility functions',
      packages=setuptools.find_packages(),
      install_requires=requirements,
     )

