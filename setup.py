from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="rrct",
    version="1.0.5",
    author="Rasheed Ibraheem",
    author_email="R.O.Ibraheem@sms.ed.ac.uk",
    maintainer="Rasheed Ibraheem",
    maintainer_email="R.O.Ibraheem@sms.ed.ac.uk",
    description="Relevance, Redundancy, and Complementarity Trade-off, a robust feature selection algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["rrct"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    license="CC-BY-4.0",
    url="https://github.com/Rasheed19/rrct-modification",
)
