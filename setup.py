from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="camelbird",
    version="0.0.1",
    description="Fair Machine Learning",
    url="https://github.com/hildeweerts/camelbird",
    author="Hilde Weerts",
    author_email="githubspam@hildeweerts.nl",
    licence="BSD 3-Clause License",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["numpy",
                      "scikit-learn",
                      ],
    python_requires='>=3.6',
)