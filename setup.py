from setuptools import setup, find_packages

setup(
    name="neuralkit",
    version="0.2.1",
    author="14k5hy4",
    author_email="",
    description="A lightweight neural network framework built from scratch in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/14k5hy4/neuralkit",
    project_urls={
        "Bug Tracker": "https://github.com/14k5hy4/neuralkit/issues",
        "Source": "https://github.com/14k5hy4/neuralkit",
        "Demo": "https://colab.research.google.com/github/14k5hy4/neuralkit/blob/master/examples/demo_notebook.ipynb",
    },
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    keywords="neural-network machine-learning deep-learning numpy from-scratch",
)
