"""Installable for reviewers: ``pip install -e ./extensions/slcd_2d/``.

The extension is a separate distribution from ``coopetition_gym`` on purpose —
v1 semantics stay frozen. This setup.py deliberately contains no entry_points,
no scripts, and no extras_require that would reach back into the base package.
"""

from setuptools import setup

setup(
    name="coopetition-gym-slcd2d",
    version="0.1.1",
    description="2D SLCD (cooperation + appropriation) extension for coopetition_gym",
    author="Vik Pant, Eric Yu",
    packages=["slcd_2d"],
    package_dir={"slcd_2d": "."},
    python_requires=">=3.10",
    install_requires=[
        "coopetition-gym>=1.0.8",
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
