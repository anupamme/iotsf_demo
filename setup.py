"""Setup configuration for the IoT Security Demo package."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="iot-security-demo",
    version="1.0.0",
    description="IoT Security Demo with Diffusion-TS and Moirai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anupam Mediratta",
    url="https://github.com/anupamme/iotsf_demo",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.29.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "plotly>=5.18.0",
        "pyyaml>=6.0.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
