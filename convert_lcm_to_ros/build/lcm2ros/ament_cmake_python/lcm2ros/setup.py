from setuptools import find_packages
from setuptools import setup

setup(
    name='lcm2ros',
    version='0.0.0',
    packages=find_packages(
        include=('lcm2ros', 'lcm2ros.*')),
)
