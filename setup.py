from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shapenet",
    version="1.0.0",
    description="Real-time shape recognition using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="punyamodi",
    url="https://github.com/punyamodi/Shape-Detection-with-CNN",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "shapenet-train=scripts.train:main",
            "shapenet-eval=scripts.evaluate:main",
            "shapenet-predict=scripts.predict:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
