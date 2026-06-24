from setuptools import setup, find_packages

setup(
    name="neuralkit",
    version="0.1.0",
    author="",
    description="A lightweight neural network toolkit built from scratch in Python",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)
