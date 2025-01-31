# Using Stochastic Gradient Descent to Smooth Nonconvex Functions: Analysis of Implicit Graduated Optimization
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for image classification.

# Abstract
The graduated optimization approach is a method for finding global optimal solutions for nonconvex functions by using a function smoothing operation with stochastic noise. We show that stochastic noise in stochastic gradient descent (SGD) has the effect of smoothing the objective function, the degree of which is determined by the learning rate, batch size, and variance of the stochastic gradient. Using this finding, we propose and analyze a new graduated optimization algorithm that varies the degree of smoothing by varying the learning rate and batch size, and provide experimental results on image classification tasks with ResNets that support our theoretical findings. We further show that there is an interesting relationship between the degree of smoothing by SGD's stochastic noise, the well-studied ``sharpness'' indicator, and the generalization performance of the model.

# Downloads
・[ImageNet dataset](https://image-net.org/index.php)  

# Wandb Setup
Please change entity name `XXXXXX` to your wandb entitiy.
```
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
```

# Usage
Please select method.
```
parser.add_argument('--method', default="batch", type=str, help="constant, lr, batch, hybrid")
```
 - "constant" means constant learning rate and constant batch size used in training.
 - "lr" means decaying leraning rate and constant batch size used in training.
 - "batch" means constant learning rate and increasing batch size used in training.
 - "hybrid" means decaying learning rate and increasing batch size used in training.

Training on CIFAR100 dataset.
```
python3 cifar.py
```
Training on ImageNet dataset.
```
python3 imagenet.py
```
