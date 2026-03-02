# Expertiential State Space

This repository contains the scripts used to optain all results resported in:

- The Geometry of Thought (under review).

The code implements the construction of an experiential state space using Principal Component Analysis (PCA), projects observations into this
space, and performs downstream statistical modelling and state-space analyses.

## Reproducibility

### Python environment

Python dependencies are listed in requirements.txt.
Create a virtual environment and install packages with:

    pip install -r requirements.txt

### R environment

This project uses renv for dependency management.
To reproduce the environment:

    install.packages("renv")  
    renv::restore()
