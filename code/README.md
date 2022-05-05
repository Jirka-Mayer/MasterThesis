# Experiments - Semi-supervised Learning in OMR

This folder contains the codebase behind my master thesis *Semi-supervised learning in Optical Music Recognition*.

The code can be run with the following packages:

```
Package                  Version
------------------------ ---------
matplotlib               3.3.4
mung                     1.1
numpy                    1.19.5
opencv-python            4.5.3.56
scikit-image             0.17.2
scipy                    1.5.4
tensorflow               2.6.0
tensorflow-datasets      4.5.2
tensorflow-probability   0.14.1
tqdm                     4.63.0
```

The code draws data from a home directory `~/Datasets`, and this directory needs to be setup for the code to work. The structure is following:

```
    ~/Datasets/
        CvcMuscima_StaffRemoval/
            CvcMuscima-Distortions/
                ideal/
                    w-01/
                    w-02/
                    ...
                    <the downloaded CVC-MUSCIMA dataset>
        DeepScoresV2/
            ds2_dense/
                images/
                instance/
                segmentation/
                deepscores_test.json
                deepscores_train.json
                <the downloaded DeepScores V2 dataset>
        MuscimaPlusPlus/
            v2.0/
                data/
                specifications/
                <the downloaded MUSCIMA++ 2.1 dataset>
```

The placement of these files can be changed in `app/datasets/constants.py`.

The code is organized into "experiments", which are large standalone modules that execute some logic. These do not correspond with experiments in the thesis, all of those are the result of the `unet` experiment. There is also the `datasets` experiment, that is designed for inspecting the dataset pipline. All other experiments are legacy and have not been used to produce any charts in the thesis.

Experiments are exposed via command-line interface, run the following to get more information:

```sh
# list available experiments
python3 main.py

# display help for an experiment
python3 main.py unet --help
```

Experiments are present in the folder `app/experiments`. For more information read their source code.

When running, experiments store results inside a folder called `experiments-data`.
