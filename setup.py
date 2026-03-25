"""
Setup script for ASL Recognition System

This allows the package to be installed in development mode:
    pip install -e .

This makes imports work cleanly throughout the project.
"""

from setuptools import setup, find_packages

setup(
    name="asl_recognition",
    version="0.1.0",
    description="Real-Time American Sign Language Recognition System",
    author="Tresa Joby",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "opencv-python>=4.8.1.78",
        "mediapipe>=0.10.8",
        "tensorflow>=2.15.0",
        "keras>=2.15.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.2",
        "tqdm>=4.66.1",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.6",
        ],
        "web": [
            "streamlit>=1.28.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
