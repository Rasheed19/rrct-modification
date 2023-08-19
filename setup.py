from setuptools import setup, find_packages

setup(
    name="rrct",
    version="1.0.2",
    author="Rasheed Ibraheem",
    author_email="R.O.Ibraheem@sms.ed.ac.uk",
    maintainer="Rasheed Ibraheem",
    maintainer_email="R.O.Ibraheem@sms.ed.ac.uk",
    description="Relevance, Redundancy, and Complementarity Trade-off, a robust feature selection algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rasheed19/rrct-modification",
    project_urls={
        "Bug Tracker": "https://github.com/Rasheed19/rrct-modification/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'pingouin',
        'openpyxl'
    ]
)
