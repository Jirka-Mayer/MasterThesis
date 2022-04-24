Chapters of the thesis and their content:

## Introduction

- OMR is being advanced using deep learning
- these models require training data, which is scarce
- producing unlabeled data is much easier
    - moreover CVC-MUSCIMA (1K pages) has only (140 pages) (14%) annotated as MPP
- semi-supervised learning approaches attempt to use the unlabeled data
    - and have been used in other fields
- we want to explore semi-supervised learning -- how much can it help here?
- music recognition has multiple stages, we will focus on object detection and recognition in the form of semantic segmentation
    - work has been done in this domain, e.g. Hajič jr., U-Net archtiecture
- there are many semi-supervised approaches, we chose to explore generative models
    - generative models attempt to learn abstract representations that help with the classification task
- we extended the U-Net architecture to make it SS compatible
- we found that improvements can be achieved, however they are relatively minor and occur in only very specific circumstances
    - this is probably because the model learns only low-level features (as seen in denoising visualizations); should it learn higher-level featuers, it could work better (GAN, etc..)
- code is on github, link


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
- solving stability (dataset seed) when increasing unsupervised ratio (fixing sup split, growing unsup split)

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


------------------

- semi-supervised improvements
    - muscima-only improvement with low supervised count on noteheads
        - vary sup/unsup pages and show improvements when
            - symbols are not "too easy"
            - there is very little supervised data
                - the model wont even converge on it (due to dropout?)

They show improvements?
https://proceedings.neurips.cc/paper/2014/file/d523773c6b194f37b938d340d5d02232-Paper.pdf

- understanding hyperparameters
    - batch size
        - too small -> bad
    - dropout
        - stabilizes training (makes it less jittery, model can be larger and wont overfit)
    - skip connections (none / gated / permanent)
        - ?
    - unsupervised loss weight
        - ?
    - adding noise during reconstruction
        - guess it does nothing, gated skipconns are more important?

- activation function
    - relu -> elu
    - relu gets stuck at the beginning of training
        - gray image to black -> overshoots -> cannot recover
            - "dying relu" problem -> cite
        - elu can
        - BUT only when training in the supervised mode -> unsup helps stabilize training
    - hajic jr. also has elu (look for the original paper and comapre)
    - the link i have from the source code uses relu, check them as well
    - performance difference - none, see the training charts

------------------


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
            implement noise generation with max noise size 2ss and 25% dropout
STEP 1: implement early stopping with grace period 10 epochs
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
- deepscores to muscima++ <--- this one!

> Experiment `knowledge-transfer`

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

- the improvement is minor and difficult to achieve
- unsup data does provide regularization
    - less noisy learning curve
    - SS coverges, when fully supervised does not
    - prevents overfitting (expand on what that means)

From introduction:
- we found that improvements can be achieved, however they are relatively minor and occur in only very specific circumstances
    - this is probably because the model learns only low-level features (as seen in denoising visualizations); should it learn higher-level featuers, it could work better (GAN, etc..)
        - show visualizations, desribe in detail
        - describe GAN as a learned-loss function


## Bibliography
