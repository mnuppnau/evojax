# Copyright 2022 The EvoJAX Authors. Licensed under Apache-2.0.
"""Packaging for the reduced research fork, not the upstream EvoJAX API."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
version = {}
exec((ROOT / "evojax/version.py").read_text(encoding="utf-8"), version)

setup(
    name="evojax",
    version=version["__version__"],
    description="PGPE and HyperNetwork research core for evolutionary hard attention",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/mnuppnau/evojax",
    license="Apache-2.0",
    packages=find_packages(include=["evojax", "evojax.*"]),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "jax==0.4.31",
        "jaxlib==0.4.31",
        "flax==0.8.4",
        "optax==0.2.4",
        "numpy==2.1.3",
    ],
    extras_require={"paper": ["matplotlib>=3.8,<4"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
    ],
)
