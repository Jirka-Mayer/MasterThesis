This experiment measures the performance of training various model capacities on the full  combined CVC-MUSCIMA and MUSCIMA++ dataset.

The commands to compute the values:

```bash
# fully supervised models
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 0 --epochs 50 --inner_features 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 0 --epochs 50 --inner_features 2
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 0 --epochs 50 --inner_features 4
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 0 --epochs 50 --inner_features 8
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 0 --epochs 50 --inner_features 16
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 0 --epochs 50 --inner_features 32

# semi-supervised models
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 551 --epochs 50 --inner_features 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 551 --epochs 50 --inner_features 2
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 551 --epochs 50 --inner_features 4
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 551 --epochs 50 --inner_features 8
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 551 --epochs 50 --inner_features 16 --seed 1 # different seed to converge
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset improvement --val_pages 20 --sup_pages 99 --unsup_pages 551 --epochs 50 --inner_features 32 --seed 1 # different seed to converge
```
