"""A setuptools based setup module.

Usage:
    pip3 install --upgrade setuptools
    sudo python3 setup.py install
"""

from setuptools import setup, find_packages

setup(
    name='NueralHJB',
    version='1.0.0',

    packages=find_packages(include=['neural_hjb', 'neural_hjb.*']),

    python_requires=">=3.7, <4",

    install_requires=[
        'hessQuik',
        'pandas',
        'numpy',
        'matplotlib',
        'torch'
    ]
)