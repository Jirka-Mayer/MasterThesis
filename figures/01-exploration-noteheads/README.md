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
| Sup | Unsup | F1  s0 | F1  s1 |
| --- | ----- | ------ | ------ |
| 10  | 0     | 92.73% | ?% |
| 10  | 5     | 90.87% | ?% |
| 10  | 10    | 91.08% | ?% |
| 10  | 50    | 92.38% | ?% |
