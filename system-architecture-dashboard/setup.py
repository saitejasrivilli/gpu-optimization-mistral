#!/usr/bin/env python3
"""System Architecture Dashboard - Production Package Setup"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="system-architecture-dashboard",
    version="1.0.0",
    author="ML Ops Team",
    author_email="mlops@example.com",
    description="Interactive dashboard for clinical ML pipeline on GCP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/system-architecture-dashboard",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9+",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "flask>=2.0.0",
        "flask-cors>=3.0.10",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dashboard-server=backend.app:main",
        ],
    },
)
