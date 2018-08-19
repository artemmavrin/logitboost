from os import path

from setuptools import setup, find_packages

import logitboost


def load_file(filename):
    filename = path.join(path.abspath(path.dirname(__file__)), filename)
    with open(filename, "r") as file:
        return file.read()


setup(
    name="logitboost",
    version=logitboost.__version__,
    description="The LogitBoost Classification Algorithm",
    long_description=load_file("README.rst"),
    long_description_content_type="text/x-rst",
    url="https://github.com/artemmavrin/logitboost",
    author="Artem Mavrin",
    author_email="amavrin@ucsd.edu",
    packages=sorted(find_packages()),
    include_package_data=True,
    install_requires=["numpy", "sklearn"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False
)
