# WaterTAP r-ENRTL

Welcome to the code repository for **WaterTAP r-ENRTL**!

![GitHub issues](https://img.shields.io/github/issues/watertap-org/watertap-renrtl)
![GitHub pull requests](https://img.shields.io/github/issues-pr/watertap-org/watertap-renrtl)
![CI status](https://img.shields.io/github/workflow/status/watertap-org/watertap-renrtl/Checks)

## Getting started (for Contributors)

**WaterTAP r-ENRTL** supports Python versions 3.8 through 3.10.

### Prerequisites

- The conda package and environment manager, for example by using the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#miniconda) following the steps appropriate for your operating system

### Installation

To install **WaterTAP r-ENRTL**, run:

```sh
git clone https://github.com/watertap-org/watertap-renrtl && cd watertap-renrtl
conda create --yes --name watertap-renrtl-dev-env python=3.10 && conda activate watertap-renrtl-dev-env
pip install -r requirements-dev.txt
```

### Running tests

```sh
conda activate watertap-renrtl-dev-env
pytest --pyargs watertap_contrib.rENRTL
```

### Formatting code

Before committing, the Python code must be formatted with [Black](https://black.readthedocs.io).

Black is installed by default as part of the developer dependencies. To format the code, run the following command from the local repository root directory:

```sh
conda activate watertap-renrtl-dev-env
black .
```
