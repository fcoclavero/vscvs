import setuptools

# tell setuptools about the package

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vscvs",
    version="0.0.1",
    author="Francisco Clavero",
    author_email="fcoclavero32@gmail.com",
    description="GAN arquitecture for creating a sketch/image common vector space with semantic information.",
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