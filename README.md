[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19058024.svg)](https://doi.org/10.5281/zenodo.19058024)

# chunked-ed

A Python package for chunked Energy Distance computation and permutation-based significance testing for 1D and 2D samples.

## Why this package?

This package was developed to make Energy Distance estimation more memory-efficient by computing pairwise distances in blocks rather than all at once.

## Features

- Energy Distance for 1D samples
- Energy Distance for 2D samples
- Chunked pairwise computation for reduced memory pressure
- Permutation-based p-value estimation
- Reproducible results with user-defined random seed

## Citation

If you use this software, please cite this release:

Akanni, I. A. (2026). *chunked-ed* (v0.1.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19058025

## Installation

### Local install

```bash
pip install .
```

### Install directly from GitHub

```bash
pip install git+https://github.com/REBELABS/chunked-ed.git
```

## Quick example

```python
import numpy as np
from chunked_ed import energy_distance, ed_p_value

a = np.random.normal(0, 1, 100)
b = np.random.normal(0.5, 1, 100)

ed = energy_distance(a, b, block=100)
print(ed)

result = ed_p_value(a, b, block=100, n_perm=200, seed=42)
print(result[0], result[1])
```