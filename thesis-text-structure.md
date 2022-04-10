Chapters of the thesis and their content:

## Introduction


## Related Work


## Current State of OMR


## Semi-supervised Learning


## Experiments and Results

### Architecture

- Denoising U-Net
- multiclass options
    - output channels
    - multiple decoders
    - seems not to improve, rather worsen if used incorrectly \[hajic\]

### Datasets

- MUSCIMA++
- DeepScores
    - https://arxiv.org/pdf/1804.00525.pdf
    - v2: https://ieeexplore.ieee.org/document/9412290/
- solving resolution problems

### Noise Generation

- noise generation and parameters

### Training

- composite batches
- loss function
- pick the model with the lowest validation loss over a training session

### Evaluation Metrics

- F1 score
- pixelwise vs. object detection
    - reference other works and their approach
    - pixelwise isn't directly telling about object detection performance
- thresholding due to varying image resolution

- object detection metrics overview:
    https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b

### Setting Hyperparameters

> guess a default and then try varying it around
> measure with respect to F1 score on a testing dataset

- batch size
    - each batch should contain both supervised and unsupervised data
    - large batches slow down training
    - should be small for fast training and also according to U-Net proposing paper (TODO: check that)
    - guess = 16 -> works even for 1:10 supervision ratio
- model hyperparameters
    - take what other works use and don't mock with it too much
    - have a reduced version for debugging?
- noise parameters
    - ???
        - set sensible defaults
    - dropout ratio 50% or 25%
    - max noise size 2ss, 1ss, 0.5ss
- unsupervised loss weight
    - 8, 4, 1, 0.25, 0.125

> Experiment `hyperparam-search`

STEP 0: use batch size 16, full resolution, ignore test set, fix seed, on noteheads,
            implement noise generation with max noise size 1ss and 25% dropout
STEP 1: implement early stopping with grace period like 10 epochs
STEP 2: write the best f1 validation score into DONE.txt file for each run
STEP 3: do a grid search over *supervision ratio* cross *unsup loss weight*
    (to get a feel for their interplay) (5x5 = 25)
STEP 4: do then a grid search of noise parameters around the optimum
    (to get a feel for the noise impact) (2x3 = 6)
STEP AGGREGATE: pulls all computed values from DONE.txt into a single composite csv file
STEP PLOTTING: select x axis and fix other values and let it plot everything that's computed

### Verifying Semi-supervised Improvements

- show how performance increases with more unsupervised data
- even for various supervised splits

> Experiment `semisup-improvements`

an in-depth exploration of the above, with optimal hyperparameters set,
for all symbol classes!


### Improving MUSCIMA++ results by utilizing CVC-MUSCIMA

- sup: mpp, unsup: cvc-m, eval: writer-independent mpp
- compare to supervised baseline
- compare to Hajič

> Experiment `improving-mpp-with-cvcm`


### Experimenting with Knowledge Transfer

- typical setup -> we have some labeled data, but it doesn't represent
    the world out there, so we add unsupervised data from the world and
    measure the segmentation on the world (was the segmentation transfered?)
- muscima++ to deepscores
- deepscores to muscima++

### Comparison to Other Works

object-detection:
- Towards Full-Pipeline handwritten OMR with musical symbol detection with U-Nets
    - fig. 4, but it's symbol detection f1-score, not pixelwise
    - implement object detection metric to compare ourselves
    - comment on the clef convex-hull hack

pixel-wise:
- Staff-line removal with selectional auto-encoders
    - https://rua.ua.es/dspace/bitstream/10045/68971/5/2017_Gallego_Calvo_ESWA_preprint.pdf
    transfer staffline removal learned supervised on deepscores and unsupervised on muscima and compare (can we learn staffline removal from printed music with unlabeled handwritten music?)

## Conclusion and Future Work


## Bibliography
