from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="autorew",
    version='0.1',
    description="A package twinturbo experiments",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Ivan Oleksiyuk",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    dependency_links=[],
)
