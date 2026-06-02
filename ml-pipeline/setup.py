from setuptools import setup, find_packages

setup(
    name="ml-pipeline",
    version="1.0.0",
    description="End-to-end ML infrastructure platform",
    author="ML Engineer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "prometheus-client>=0.19.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
