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
        "click",
        "coloredlogs",
        "h5py",
        "hdbscan",
        "imageio",
        "matplotlib",
        "nibabel",
        "numpy",
        "pytorch",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "simpleitk",  # dependency of nibabel
        "torchvision",
        "tensorboardx",
        "tqdm",
    ],
    zip_safe=True,
    extras_require={},
    entry_points={"console_scripts": ["repbuild=pattern.cli.reptools:repbuild"]},
)
