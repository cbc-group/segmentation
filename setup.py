"""
Minimal setup.py to simplify project setup.
"""
from setuptools import find_packages, setup

setup(
    name="segmentation",
    version="0.1",
    description="",
    author="Liu, Yen-Ting",
    author_email="ytliu@gate.sinica.edu.tw",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={"": ["data/*"]},
    python_requires="~=3.7",
    install_requires=[
        "click",
        "coloredlogs",
        "dask~=2.12.0",
        "distributed~=2.12.0",
        "h5py",
        "imageio",
        "numpy",
        "prefect",
        "simpleitk",  # dependency of nibabel
        "utoolbox~=0.6.0",
    ],
    zip_safe=True,
    extras_require={},
    entry_points={
        "console_scripts": [
            "bin4=segmentation.pipeline.flows.bin4:main",
            "pack=segmentation.pipeline.flows.pack:main",
            "inference=segmentation.pipeline.flows.inference:main",
            "aslabel=segmentation.pipeline.flows.aslabel:main",
        ]
    },
)
