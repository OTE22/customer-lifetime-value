#!/usr/bin/env python3
"""
CLV Prediction System - Setup Script
Author: Ali Abbass (OTE22)

For modern Python packaging, use pyproject.toml instead.
This file is provided for backwards compatibility.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ]

setup(
    name="clv-prediction-system",
    version="2.0.0",
    author="Ali Abbass",
    author_email="ali.abbass@ote22.dev",
    maintainer="Ali Abbass (OTE22)",
    maintainer_email="ali.abbass@ote22.dev",
    description="Production-ready Customer Lifetime Value Prediction System with ML and API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OTE22/customer-lifetime-value",
    project_urls={
        "Bug Tracker": "https://github.com/OTE22/customer-lifetime-value/issues",
        "Documentation": "https://github.com/OTE22/customer-lifetime-value#readme",
        "Source Code": "https://github.com/OTE22/customer-lifetime-value",
    },
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_data={
        "backend": ["*.json", "*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "redis": [
            "redis>=4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clv-api=backend.api_enhanced:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Typing :: Typed",
    ],
    keywords=[
        "machine-learning",
        "customer-lifetime-value",
        "clv",
        "prediction",
        "e-commerce",
        "fastapi",
        "meta-ads",
        "marketing",
        "analytics",
    ],
)
