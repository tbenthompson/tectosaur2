#!/bin/zsh
eval "$(conda shell.zsh hook)"
conda activate tectosaur2
python setup.py build_ext --inplace
