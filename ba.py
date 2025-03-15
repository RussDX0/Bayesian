# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:13:00 2025

@author: Russell Talarion
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic delivery time data
np.random.seed(176)
true_mu = 4.5  # True average delivery time in days
true_sigma = 1.2  # True standard deviation
data = np.random.normal(true_mu, true_sigma, size=100)  # Simulated delivery times

# Define the prior hyperparameters
prior_mu_mean = 3  # Prior belief: 3 days average delivery time
prior_mu_precision = 0.5  # Low confidence in prior (high variance)
prior_sigma_alpha = 2  # Weak prior for sigma
prior_sigma_beta = 1  # Beta = alpha / beta

# Update the prior hyperparameters with the data
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Calculate the posterior distributions
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plot the posterior distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior Distribution of Average Delivery Time (μ)')
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior Distribution of Delivery Time Variability (σ)')
plt.xlabel('Standard Deviation (Days)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Calculate summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Estimated Mean Delivery Time:", mean_mu)
print("Standard Deviation of Estimated Mean:", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Estimated Standard Deviation of Delivery Time:", mean_sigma)
print("Standard Deviation of Estimated Sigma:", std_sigma)