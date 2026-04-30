"""Installable for reviewers: ``pip install -e ./extensions/slcd_2d/``.

The extension is a separate distribution from ``coopetition_gym`` on purpose —
v1 semantics stay frozen. This setup.py deliberately contains no entry_points,
no scripts, and no extras_require that would reach back into the base package.
"""

from setuptools import setup, find_packages

setup(
    name="coopetition-gym-slcd2d",
    version="0.1.0",
    description="2D SLCD (cooperation + appropriation) extension for coopetition_gym",
    author="Vik Pant, Eric Yu",
    packages=find_packages(where=".", include=["slcd_2d*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "coopetition-gym>=0.2.0",
        "numpy>=1.22",
        "scipy>=1.10",
        "gymnasium>=0.29",
    ],
    extras_require={
        "training": ["stable-baselines3>=2.0", "torch>=2.0"],
        "dev": ["pytest>=7"],
    },
    include_package_data=True,
    package_data={"slcd_2d": ["calibration.json"]},
)
