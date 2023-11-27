import sys
from setuptools import find_packages, setup


if sys.version_info < (3, 9):
    sys.exit("Requires Python 3.9 or higher")

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_ = f.read()

REQUIRED_PACKAGES = [
    "SimpleITK",
    "pydicom",
]

TEST_PACKAGES = []

setup(
    name="dosemetrics",
    version="0.1",
    description="Measuring radiotherapy doses - plotting visualizations",
    long_description=readme,
    author="Amith Kamath",
    author_email="amith.kamath@unibe.ch",
    url="https://github.com/amithjkamath/dosemetrics",
    license=license_,
    packages=find_packages(exclude=["test", "docs"]),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=["medical image analysis", "machine learning", "neuro"],
)
