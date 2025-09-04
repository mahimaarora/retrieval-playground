from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def read_requirements():
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        return [
            line.strip() for line in requirements_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []

setup(
    name="retrieval-playground",
    version="0.1.0",
    author="Mahima Arora",
    author_email="maharora@redhat.com",
    description="A playground for retrieval-augmented generation (RAG) experiments",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "rp-generate-test-data=retrieval_playground.tests.test_data_generation:cli_main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8+",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
