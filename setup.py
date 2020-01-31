"""
Minimal setup.py to simplify project setup.
"""
from setuptools import find_packages, setup

setup(
    name="stitching",
    version="0.1",
    description="",
    author="Liu, Yen-Ting",
    author_email="ytliu@gate.sinica.edu.tw",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={"": ["data/*"]},
    install_requires=[
        "coloredlogs",
        "h5py",
        "imageio",
        "numpy",
        "simpleitk",  # dependency of nibabel
    ],
    zip_safe=True,
    extras_require={},
    entry_points={"console_scripts": []},
)
