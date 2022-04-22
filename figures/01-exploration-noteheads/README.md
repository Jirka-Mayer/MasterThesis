This experiment measures the impact of changing the labeled to unlabeled data ratio, when doing notehead segmentation.

The commands to compute these values:

```bash
# 0% dropout, 100 epochs
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 50 --epochs 100
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 10 --epochs 100
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 5 --epochs 100
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 0 --epochs 100

# 50% dropout, 200 epochs
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 50 --dropout 0.5 --epochs 200
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 10 --dropout 0.5 --epochs 200
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 5 --dropout 0.5 --epochs 200
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 0 --dropout 0.5 --epochs 200

# 50% dropout, 200 epochs, seed 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 50 --dropout 0.5 --epochs 200 --seed 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 10 --dropout 0.5 --epochs 200 --seed 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 5 --dropout 0.5 --epochs 200 --seed 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --unsup_pages 0 --dropout 0.5 --epochs 200 --seed 1
```

MUSCIMA++ test set results:

**without dropout**
| Sup | Unsup | F1  s0 |
| --- | ----- | ------ |
| 10  | 0     | 91.25% |
| 10  | 5     | 88.80% |
| 10  | 10    | 89.37% |
| 10  | 50    | 90.29% |

**with dropout**
| Sup | Unsup | F1  s0 | F1  s1 | F1  s2 | F1  s3 | F1  s4 | F1  s5 |
| --- | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
| 10  | 0     | 92.73% | 93.02% | 93.54% | 93.34% | 93.90% | 93.00% |
| 10  | 5     | 90.87% | 90.36% | 90.05% | 90.99% | 90.58% | 90.54% |
| 10  | 10    | 91.08% | 90.13% | 89.88% | 89.65% | 90.99% | 91.43% |
| 10  | 50    | 92.38% | 91.24% | 91.56% | 92.63% | 90.95% | 92.37% |
