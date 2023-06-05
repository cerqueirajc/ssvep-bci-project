from setuptools import setup, find_packages  # or find_namespace_packages

setup(
    name="mymodel",
    version="0.0.2",
    install_requires=[
        "numpy"
    ],
    packages=find_packages(
        where='src/ssvep-bci-project',
    ),
)