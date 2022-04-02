# Semi-supervised learning in Optical Music Recognition

## Playground

See the `playground` folder.


## Notes

An Auto-Encoder Strategy for Adaptive Image Segmentation:
https://arxiv.org/pdf/2004.13903.pdf

Gated Feedback Refinement Network for Coarse-to-Fine Dense Semantic Image Labeling:
https://arxiv.org/pdf/1806.11266.pdf

An Overview of Deep Semi-Supervised Learning
https://arxiv.org/pdf/2006.05278.pdf

Few-Shot semantic segmentation papers
https://github.com/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers

Stacked Convolutional Sparse Auto-Encoders for Representation Learning
https://dl.acm.org/doi/abs/10.1145/3434767

Sparse autoencoder: Lecture notes (Andrew Ng)
https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf

Awesome list:
https://github.com/yassouali/awesome-semi-supervised-learning

Understanding the Effective Receptive Field in Deep Convolutional Neural Networks
https://arxiv.org/pdf/1701.04128.pdf

Dynamic Routing Between Capsules
https://cmp.felk.cvut.cz/~toliageo/rg/papers/SabourFrosstHinton_NIPS2017_Dynamic%20Routing%20Between%20Capsules.pdf


## The plan

```
- data feeding pipeline
    - muscima++
        - MuscimaPage -> binary image
        - MuscimaPage, mask classes -> mask image
    - utilities
        - tile sampling
        - image resizing
    - noise
        - any image -> noisy image, noise mask
        - noise parameters
            - optimize for best "feature extraction"
                = maximum metric improvement for given semi-sup split
    - deepscores
        - ?
- model API
    - (labeled_x, unlabeled_x) -> (labeled_y, unlabeled_y)
    - parameters
        - number of input/output channels
        - model-specific hyperparameters
- possible models
    - denoising U-Net

Testing jig:
- When training
    1) select training and validation datasets
    2) select model with all parameters, compatible with the datasets
- When evaluating
    1) select evaluation dataset
    2) select trained model
- Possible other experiments with trained models
- Model + training dataset + options = setup
```

Questions to answer:

- What unlabeled data ratio is the best?
- What noise pattern is the best?
- Can the model generalize from muscima++ to deepscores?

Evaluation:

- Best = best F1 score on evaluation dataset
- evaluation datasets
    - muscima++ writer-independent test set
    - deepscores subset?

Symbols important for muscic recognition:

    noteheads (2x)
    accidentals (3x)
    stem
    beam
    flag
    rest (4x)
    ledger line
    barline
    slur
    duration dot
    staccato dot
