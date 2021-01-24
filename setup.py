import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlb-core",
    version="0.0.4",
    author="Matthew Bowers",
    author_email="mattb6503@gmail.com",
    description="a custom suite of utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlb2251/mlb",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

