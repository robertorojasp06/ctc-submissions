#!/bin/bash
# create conda environment. If already exists, it is updated
conda env create -f environment.yml || conda env update -f environment.yml