#!/bin/bash

cd ~/Datasets

wget https://zenodo.org/record/4012193/files/ds2_dense.tar.gz

mkdir DeepScoresV2
tar -xf ds2_dense.tar.gz
mv ds2_dense DeepScoresV2/ds2_dense
