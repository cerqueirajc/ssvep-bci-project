from setuptools import setup, find_packages  # or find_namespace_packages

setup(
    name="ssvepcca",
    version="0.0.1",
    author="Joao Cerqueira",
    author_email="jc.cerqueira13@gmail.com",
    description = (
        "Project to experiment with different algorithms for "
        "recognizing the frequency of steady-state visual "
        "evoked potentials (SSVEP) in electroencephalogram (EEG)"
    ),
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "pandas"
    ],
)