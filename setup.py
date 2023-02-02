from setuptools import setup, find_namespace_packages

setup(
    name="watertap-renrtl",
    version="0.1.dev0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    author="WaterTAP r-ENRTL contributors",
    python_requires=">=3.8",
    install_requires=[
        "watertap >= 0.6.0",
        "pytest >= 7",
    ],
)
