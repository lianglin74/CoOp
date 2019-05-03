from setuptools import setup

# suggest to install by
# python setup.py build develop --user

setup(name='qd',
      version='0.1',
      description='Provide basic utility functions',
      packages=['qd'],
     )

setup(name='qd_classifier',
      version='0.1',
      description='Training code for the 2nd stage classifier after detection',
      packages=['qd_classifier'],
     )
