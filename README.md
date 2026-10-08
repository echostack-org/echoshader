# Welcome to echoshader

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/echostack-org/echoshader/HEAD)

Open Source Python package for building ocean sonar data visualizations based on the HoloViz suite of Python tools.

## What are ocean sonar systems?

Ocean sonar systems, such as echosounders, are the [workhorse to study life in the ocean](https://storymaps.arcgis.com/stories/e245977def474bdba60952f30576908f). They provide continuous observations of fish and zooplankton by transmitting sounds and analyzing the echoes bounced off these animals, just like how medical ultrasound images the interior of the human body. In recent years these systems have been widely deployed on ships, autonomous vehicles, and moorings, bringing in large volumes of data that allow scientists to study changes in marine ecosystems.

## What is echoshader?

Echoshader aims to enhance the capability to interactively visualize large volumes of ocean sonar data to accelerate data exploration and discovery. The
project goes hand-in-hand with the ongoing development of
[echopype](https://echopype.readthedocs.io/en/stable/), which handles the
standardization, pre-processing, and organization of these data.

By providing an accessible and customizable platform for echo data visualization, the project can accelerate advancements in oceanographic research for the benefit of conservation and sustainable resource management.

## Installation

Echoshader supports Python 3.13 and 3.14.
We recommend creating a new environment before installing echoshader:

```bash
mamba create -c conda-forge -n echoshader --yes python=3.13
mamba activate echoshader
```

To install the latest release from PyPI:

```bash
python -m pip install echoshader
```

To install the latest development version from GitHub:

```bash
python -m pip install git+https://github.com/echostack-org/echoshader.git
```

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) to manage conda environments.

## Development setup

This section is intended for those who are actively developing echoshader.

First, fork and clone the [echoshader repository](https://github.com/echostack-org/echoshader), then create and activate a development environment:

```bash
mamba create -c conda-forge -n echoshader-dev --yes python=3.13
mamba activate echoshader-dev
```

From the repository directory, install echoshader in editable mode together with the development and testing dependencies defined in `pyproject.toml`:

```bash
python -m pip install -e ".[dev,test]"
```

To link the environment with a Jupyter kernel:

```bash
python -m ipykernel install --user \
    --name echoshader-dev \
    --display-name "echoshader-dev"
```

To run the test suite:

```bash
pytest
```
