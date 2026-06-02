from setuptools import setup, find_packages

setup(
    name="mini-spark",
    version="1.0.0",
    description="Distributed batch processing engine",
    author="ML Engineer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "redis>=5.0.0",
        "protobuf>=4.25.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
