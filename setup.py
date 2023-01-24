from setuptools import setup

with open("requirements.txt") as file:
    requirements = file.readlines()

setup(
    name='src',
    version='0.0.1',
    description="Python library for abstracting feed-forward neural networks",
    author="Calvin Chau, Stefanie Mohr, Jan Křetı́nský",
    author_email="calvin.chau@tum.de",
    install_requires=requirements
)
