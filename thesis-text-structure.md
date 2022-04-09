Chapters of the thesis and their content:

## Introduction


## Related Work


## Current State of OMR


## Semi-supervised Learning


## Experiments and Results

### Architecture

- Denoising U-Net

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
- unsupervised loss weight
    - ???
- noise parameters
    - ???
- model hyperparameters
    - ???

### Verifying Semi-supervised Improvements

- show how performance increases with more unsupervised data
- even for various supervised splits

> Experiment `semisup-improvements`


### Multiclass Segmentation Effect on Performance

- does adding more classes affect performance of individual classes?

> Experiment `multiclass-effects`


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
