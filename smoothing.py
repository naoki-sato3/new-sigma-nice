import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy

#Rastrigin's function
def f(x):
    return x**2 - 10 * np.cos(2 * np.pi * x) + 10

#Drop-Wave function
def f(x):
    numerator = -1 - np.cos(12 * np.pi * x)
    denominator = 2 + 0.5 * x**2
    return numerator / denominator

def smooth_f_gaussian(x, delta, num_samples=1000):
    u_samples = np.random.normal(0, 1, num_samples)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)

def smooth_f_uniform(x, delta, num_samples=1000, a=-2, b=2):
    u_samples = np.random.uniform(a, b, num_samples)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)

def smooth_f_exponential(x, delta, num_samples=1000, scale=1.5):
    u_samples = np.random.exponential(scale, num_samples)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)

def smooth_f_rayleigh(x, delta, num_samples=1000, scale=1.0):
    u_samples = np.random.rayleigh(scale, num_samples)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)


def smooth_f_cauchy(x, delta, num_samples=1000):
    u_samples = np.random.standard_cauchy(num_samples)
    #u_samples = np.clip(u_samples, 0, 0.5)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)

def smooth_f_levy(x, delta, num_samples=1000, c=1.0):
    u_samples = levy.rvs(c, size=num_samples)
    #u_samples = np.clip(u_samples, -5, 2)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)

def smooth_f_pareto(x, delta, num_samples=1000, a=1.0):
    u_samples = np.random.pareto(a, num_samples) + 1
    #u_samples = np.clip(u_samples, -5, 2)
    smoothed_values = [f(x - delta * u) for u in u_samples]
    return np.mean(smoothed_values)

x_values = np.linspace(-5, 5, 100)
delta = 0.5  

smoothed_f_gaussian_values = [smooth_f_gaussian(x, delta) for x in x_values]
smoothed_f_uniform_values = [smooth_f_uniform(x, delta) for x in x_values]
smoothed_f_exponential_values = [smooth_f_exponential(x, delta) for x in x_values]
smoothed_f_rayleigh_values = [smooth_f_rayleigh(x, delta) for x in x_values]
smoothed_f_cauchy_values = [smooth_f_cauchy(x, delta) for x in x_values]
smoothed_f_levy_values = [smooth_f_levy(x, delta) for x in x_values]
smoothed_f_pareto_values = [smooth_f_pareto(x, delta) for x in x_values]

plt.plot(x_values, [f(x) for x in x_values], label='Original f(x)', linestyle='--', color='gray')
plt.plot(x_values, smoothed_f_gaussian_values, label=r'$\hat{f}_{\delta}$ (Gaussian)')
plt.plot(x_values, smoothed_f_uniform_values, label=r'$\hat{f}_{\delta}$ (Uniform)')
plt.plot(x_values, smoothed_f_exponential_values, label=r'$\hat{f}_{\delta}$ (Exponential)')
plt.plot(x_values, smoothed_f_rayleigh_values, label=r'$\hat{f}_{\delta}$ (Rayleigh)')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.title('Smoothed Function $\hat{f}_{\delta}$ with light-tailed distributions')
plt.legend()
plt.savefig("light.pdf", dpi=800)
#plt.show()

plt.plot(x_values, [f(x) for x in x_values], label='Original f(x)', linestyle='--', color='gray')
plt.plot(x_values, smoothed_f_pareto_values, label=r'$\hat{f}_{\delta}$ (Pareto)', color='purple')
plt.plot(x_values, smoothed_f_cauchy_values, label=r'$\hat{f}_{\delta}$ (Cauchy)', color='pink')
plt.plot(x_values, smoothed_f_levy_values, label=r'$\hat{f}_{\delta}$ (Levy)', color='olive')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.title('Smoothed Function $\hat{f}_{\delta}$ with heavy-tailed distributions')
plt.legend()
plt.savefig("heavy.pdf", dpi=800)
#plt.show()
