__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Setuptools parameters for packaging the project. """


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vscvs",
    version="0.0.1",
    author="Francisco Clavero",
    author_email="fcoclavero32@gmail.com",
    description="GAN architecture for creating a sketch/image common vector space with semantic information.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fcoclavero/vscvs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
