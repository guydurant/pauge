from setuptools import setup, find_packages

setup(
    name="pauge",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "rich",
        "posebusters",
        "numpy",
        "pymol-open-source",
        "requests",
        "biopython",
        "pypdb",
        "openbabel-wheel",
        "click",
        "tqdm",
        "joblib",
        "molecular-rectifier",
        "plip", 
        ],
    entry_points={
        "console_scripts": [
            "pauge_classic=src.classic.cli:main",
            "pauge_plausability=src.plausability.cli:main",
        ],
    },
    python_requires=">=3.8",
)
