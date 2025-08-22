#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 17:54:13 2025

@author: sevak
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Set a random seed for reproducibility
np.random.seed(0)

# --- Generate Synthetic Data ---
n_samples = 15
X = np.sort(np.random.uniform(0, 10, size=(n_samples, 1)), axis=0)
y = X.ravel()**2 + 10 * np.cos(X.ravel()) + np.random.normal(0, 5, n_samples)

X_test = np.linspace(0, 10, 100).reshape(-1, 1)

# Polynomial degree
degree = 9

# Model 1: No Regularization
poly_no_reg = PolynomialFeatures(degree)
X_poly_no_reg = poly_no_reg.fit_transform(X)
model_no_reg = LinearRegression()
model_no_reg.fit(X_poly_no_reg, y)
y_train_pred_no_reg=model_no_reg.predict(X_poly_no_reg)
mse_train_no_reg = mean_squared_error(y, y_train_pred_no_reg)
#y_test_pred_no_reg=model_no_reg.predict(X_test)
#mse_test_no_reg = mean_squared_error(y, y_test_pred_no_reg)

# Model 2: L1 Regularization (Lasso)
poly_l1 = PolynomialFeatures(degree)
X_poly_l1 = poly_l1.fit_transform(X)
model_l1 = Lasso(alpha=10, max_iter=10000)
model_l1.fit(X_poly_l1, y)
y_train_pred_l1=model_l1.predict(X_poly_l1)
mse_train_l1 = mean_squared_error(y, y_train_pred_l1)

# Model 3: L2 Regularization (Ridge)
poly_l2 = PolynomialFeatures(degree)
X_poly_l2 = poly_l2.fit_transform(X)
model_l2 = Ridge(alpha=1000000,max_iter=10000)
model_l2.fit(X_poly_l2, y)
y_train_pred_l2=model_l2.predict(X_poly_l2)
mse_train_l2 = mean_squared_error(y, y_train_pred_l2)

print("MSE in No Regression (Training data)",mse_train_no_reg)
print("MSE in Lasso (Training data)",mse_train_l1)
print("MSE in Ridge (Training data)",mse_train_l2)

# Predictions
X_test_poly_no_reg = poly_no_reg.transform(X_test)
y_pred_no_reg = model_no_reg.predict(X_test_poly_no_reg)
#print(y_pred_no_reg)

X_test_poly_l1 = poly_l1.transform(X_test)
y_pred_l1 = model_l1.predict(X_test_poly_l1)

X_test_poly_l2 = poly_l2.transform(X_test)
y_pred_l2 = model_l2.predict(X_test_poly_l2)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
models = [(y_pred_no_reg, 'No Regularization (Overfitting)', ''),
          (y_pred_l1, 'L1 Regularization (Lasso)', ', λ = 10'),
          (y_pred_l2, 'L2 Regularization (Ridge)', ', λ = $10^6$')]

x_min, x_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

for i, (y_pred, title, alpha_val) in enumerate(models):
    ax = axes[i]
    ax.scatter(X, y, color='blue', s=60, label='Training Data (15 points)')
    ax.plot(X_test, y_pred, color='red', linewidth=3, label=f'Poly. Degree {degree} {alpha_val}')
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('X', fontsize=14)
    if i == 0:
        ax.set_ylabel('y', fontsize=14)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.legend(fontsize=12)
    ax.grid(True)

plt.show()

# Print coefficients for all models
# print("\n--- No Regularization Model Coefficients (First 10) ---")
# print(np.round(model_no_reg.intercept_, 4))
# print(np.round(model_no_reg.coef_[:10], 4))

# print("\n--- L1 Regularization Model Coefficients (First 10) ---")
# print(np.round(model_l1.intercept_, 4))
# print(np.round(model_l1.coef_[:10], 4))
# print(f"Number of non-zero coefficients: {np.sum(model_l1.coef_ != 0)}")

# print("\n--- L2 Regularization Model Coefficients (First 10) ---")
# print(np.round(model_l2.intercept_, 4))
# print(np.round(model_l2.coef_[:10], 4))

#print("MSE in No Regression (Training data)",mse_train_no_reg)
#print("MSE in Lasso (Training data)",mse_train_l1)
#print("MSE in Ridge (Training data)",mse_train_l2)
#print("MSE in No Regression (Testing data)",mse_test_no_reg)
#print("MSE in Lasso (Testing data)",mse_test_l1)
#print("MSE in Ridge (Testing data)",mse_test_l2)


