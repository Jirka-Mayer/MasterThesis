#!/bin/bash

cd ~/Datasets

# Official, but slow download URL: http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_SR.zip
wget https://github.com/apacha/OMR-Datasets/releases/download/datasets/CVCMUSCIMA_SR.zip

mkdir CvcMuscima_StaffRemoval
unzip CVCMUSCIMA_SR.zip
mv CvcMuscima-Distortions CvcMuscima_StaffRemoval/CvcMuscima-Distortions
