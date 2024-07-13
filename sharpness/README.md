# Experiments in Section 3.3.1

This implements based on [A modern look at the relationship between sharpness and generalization](https://arxiv.org/abs/2302.07011)(ICML2023).

Please see https://github.com/tml-epfl/sharpness-vs-generalization.

# Wandb Setup
Please change entity name `XXXXXX` to your wandb entitiy.
```
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
```

# Usage
Train 200 epochs on the CIFAR100 dataset and measure "sharpness".
```
python3 main.py
```

