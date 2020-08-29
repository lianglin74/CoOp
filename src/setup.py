from setuptools import setup
import setuptools

# suggest to install by
# python setup.py build develop --user

setup(name='qd',
      version='0.1',
      description='Provide basic utility functions',
      #packages=['qd', 'qd_classifier'],
      packages=setuptools.find_packages(),
     )

