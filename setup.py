"""
Setup configuration for the Human-AI Cognition Framework
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="human-ai-cognition",
    version="0.1.0",
    author="Human-AI Cognition Team",
    author_email="team@human-ai-cognition.dev",
    description="A biologically-inspired cognitive architecture for AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/human-ai-cognition/framework",
    project_urls={
        "Bug Tracker": "https://github.com/human-ai-cognition/framework/issues",
        "Documentation": "https://human-ai-cognition.readthedocs.io/",
        "Source Code": "https://github.com/human-ai-cognition/framework",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "aws": [
            "boto3>=1.28.0",
            "botocore>=1.31.0",
        ],
        "visualization": [
            "streamlit>=1.45.0",
            "plotly>=5.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "statsmodels>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cognitive-ai=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
