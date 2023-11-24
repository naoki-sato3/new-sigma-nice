# Using Stochastic Gradient Descent to Smooth Nonconvex Functions: Analysis of Implicit Graduated Optimization with Optimal Noise Scheduling
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for image classification.

# Abstract
The graduated optimization approach is a heuristic method for finding globally optimal solutions for nonconvex functions and has been theoretically analyzed in several studies. This paper defines a new family of nonconvex functions for graduated optimization, discusses their sufficient conditions, and provides a convergence analysis of the graduated optimization algorithm for them. It shows that stochastic gradient descent (SGD) with mini-batch stochastic gradients has the effect of smoothing the function, the degree of which is determined by the learning rate and batch size. This finding provides theoretical insights on why large batch sizes fall into sharp local minima, why decaying learning rates and increasing batch sizes are superior to fixed learning rates and batch sizes, and what the optimal learning rate scheduling is. To the best of our knowledge, this is the first paper to provide a theoretical explanation for these aspects. Moreover, a new graduated optimization framework that uses a decaying learning rate and increasing batch size is analyzed and experimental results of image classification that support our theoretical findings are reported.

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
parser.add_argument('--method', default="batch", type=str, help="constant, lr, batch, hybrid, poly, cosine, exp")
```
Training on CIFAR100 dataset.
```
python3 cifar.py
```
Training on ImageNet dataset.
```
python3 imagenet.py
```
