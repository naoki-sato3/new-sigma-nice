# Using Stochastic Gradient Descent to Smooth Nonconvex Functions: Analysis of Implicit Graduated Optimization with Optimal Noise Scheduling
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for image classification.

# Abstract
The graduated optimization approach is a heuristic method for finding globally optimal solutions for nonconvex functions and has been theoretically analyzed in several studies. This paper defines a new family of nonconvex functions for graduated optimization, discusses their sufficient conditions, and provides a convergence analysis of the graduated optimization algorithm for them. It shows that stochastic gradient descent (SGD) with mini-batch stochastic gradients has the effect of smoothing the objective function, the degree of which is determined by the learning rate, batch size, and variance of the stochastic gradient. This finding provides theoretical insights on why large batch sizes fall into sharp local minima, why decaying learning rates and increasing batch sizes are superior to fixed learning rates and batch sizes, and what the optimal learning rate scheduling is. To the best of our knowledge, this is the first paper to provide a theoretical explanation for these aspects. In addition, we show that the degree of smoothing introduced is strongly correlated with the generalization performance of the model. Moreover, a new graduated optimization framework that uses a decaying learning rate and increasing batch size is analyzed and experimental results of image classification are reported that support our theoretical findings.

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
parser.add_argument('--method', default="batch", type=str, help="constant, lr, batch, hybrid, poly, cosine, exp, sampling")
```
 - "constant" means constant learning rate and constant batch size used in training.
 - "lr" means decaying leraning rate and constant batch size used in training.
 - "batch" means constant learning rate and increasing batch size used in training.
 - "hybrid" means decaying learning rate and increasing batch size used in training.
 - "poly" means polynomial decay learning rate used in training.
 - "cosine" means cosine annealing used in training.
 - "exp" means exponential decay learning rate used in training.
 - "sampling" measure a stochastic noise norm (see Figure 8).

Training on CIFAR100 dataset.
```
python3 cifar.py
```
Training on ImageNet dataset.
```
python3 imagenet.py
```
