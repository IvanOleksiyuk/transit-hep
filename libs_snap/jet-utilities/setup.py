import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jet-utilities",
    version="0.0.1",
    author="Matthew Leigh, Knut Zoch, Matthias Schlaffer",
    description="Some common utilities for RODEM jet data handling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.cern.ch/geneva-anomalous-jets/jet-utilities",
    project_urls={
        "Bug Tracker": "https://gitlab.cern.ch/geneva-anomalous-jets/jet-utilities/issues"
    },
    license="MIT",
    packages=["jutils"],
    include_package_data=True,
    install_requires=["cerberus", "h5py", "numpy", "pytest", "importlib_resources"],
)
