This experiment tries to transfer knowledge of printed staffline segmentation onto handwritten stafflines, utlizing handwritten music as unsupervised training data.

The commands to compute the values:

```bash
# fully supervised models
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 0 --epochs 20 --seed 0
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 0 --epochs 20 --seed 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 0 --epochs 20 --seed 2
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 0 --epochs 20 --seed 3

# semi-supervised models
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 99 --epochs 20 --seed 0
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 99 --epochs 20 --seed 1
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 99 --epochs 20 --seed 2
~/interactive.sh ~/pyenv/bin/python3 main.py unet train --dataset transfer2mpp --val_pages 20 --sup_pages 20 --unsup_pages 99 --epochs 20 --seed 3
```
