from setuptools import setup, find_packages

setup(
    name="SWAGEN",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    author="Jonathan Karin, Zoe Piran, Mor Nitzan",
    description="SWAGEN: A framework to enhance swarm durability via GNN-based generative modeling",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nitzanlab/SwaGen",
)
