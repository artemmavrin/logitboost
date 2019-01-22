from os import path

from setuptools import setup, find_packages

import logitboost


def load_file(filename):
    filename = path.join(path.abspath(path.dirname(__file__)), filename)
    with open(filename, "r") as _file:
        return _file.read()


setup(
    name="logitboost",
    version=logitboost.__version__,
    description="The LogitBoost Classification Algorithm",
    long_description=load_file("README.rst"),
    long_description_content_type="text/x-rst",
    url="https://github.com/artemmavrin/logitboost",
    author="Artem Mavrin",
    author_email="artemvmavrin@gmail.com",
    packages=sorted(find_packages(exclude=("*.tests",))),
    include_package_data=True,
    install_requires=[
        "numpy>=1.15",
        "scipy>=1.1",
        "scikit-learn>=0.20"
    ],
    extras_require={"examples": ['jupyter', 'matplotlib', 'seaborn']},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False,
    license="MIT",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License"
    ]
)
