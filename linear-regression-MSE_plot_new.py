import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.metrics import mean_squared_error
import math

def add_dummy_feature(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def parameter_linear_regression(X,y):
	X_train_b_linear = add_dummy_feature(X)
	# Calculate the optimal theta for linear regression using the Normal Equation on training data
	theta_best_linear = np.linalg.inv(X_train_b_linear.T @ X_train_b_linear) @ X_train_b_linear.T @ y
	return theta_best_linear

def parameter_cubic_regression(X,y):
	X_train_poly = np.c_[X, X**2, X**3]
	# Add dummy feature to the polynomial training data
	X_train_poly_b = add_dummy_feature(X_train_poly)
	# Calculate the optimal theta for cubic regression using the Normal Equation on training data
	theta_best_cubic = np.linalg.inv(X_train_poly_b.T @ X_train_poly_b) @ X_train_poly_b.T @ y
	return theta_best_cubic

def polly_train_X(X,deg):
	deg_train=X
	for i in range(2,deg+1):
		deg_train=np.c_[deg_train,X**i]
	X_train_poly_new=deg_train
	X_train_poly_b_new = add_dummy_feature(X_train_poly_new)
	return X_train_poly_b_new

def parameter_polynomial_regression(X,y,deg):
	X_train_poly_b_new=polly_train_X(X,deg)
	theta_best_polly=np.linalg.inv(X_train_poly_b_new.T @ X_train_poly_b_new) @ X_train_poly_b_new.T @ y_train
	return theta_best_polly


def comperison_linear_and_cubic(X,y,theta_best_linear,theta_best_cubic):
	# --- Plotting ---
	plt.figure(figsize=(10, 7)) # Increased figure size for better visibility
	plt.plot(X, y, "b.", label="Original Data (All)") # Plot all original data points

	# To get a smooth curve for plotting the regression lines, we need more points.
	X_plot = np.linspace(0, 3, 100).reshape(-1, 1)

	# Prepare the new data points for linear prediction plotting
	X_new_b_plot_linear = add_dummy_feature(X_plot)
	# Predict using the linear model (trained on training data)
	y_predict_linear_plot = X_new_b_plot_linear @ theta_best_linear
	# Plot the linear regression line
	plt.plot(X_plot, y_predict_linear_plot, "r-", label="Linear Prediction")

	# Prepare the new data points for cubic prediction plotting
	X_plot_poly_cubic = np.c_[X_plot, X_plot**2, X_plot**3]
	X_plot_poly_b_cubic = add_dummy_feature(X_plot_poly_cubic)
	# Predict using the cubic model (trained on training data)
	y_predict_cubic_plot = X_plot_poly_b_cubic @ theta_best_cubic
	# Plot the cubic regression line
	plt.plot(X_plot, y_predict_cubic_plot, "g-", label="Cubic Prediction")

	plt.xlabel("$\mathbf{x}$", fontsize=14) # Increased font size for labels
	plt.ylabel("y", fontsize=14)

	# Get the min and max values of X and y
	x_min, x_max = X.min(), X.max()
	y_min, y_max = y.min(), y.max()

	# It's good practice to add a little padding to the axis limits
	x_padding = (x_max - x_min) * 0.1
	y_padding = (y_max - y_min) * 0.1

	# Set the axis limits with padding
	plt.axis([x_min - x_padding, x_max + x_padding, y_min - y_padding, y_max + y_padding])

	plt.grid(True) # Ensure grid is visible
	plt.legend(loc="upper left", fontsize=12) # Increased font size for legend

	plt.title("Linear vs. Cubic Regression with Train/Test Split", fontsize=16)
	plt.show()


def mse_in_various_deg (polynomial_degrees,train_errors_mse,test_errors_mse):
	plt.figure(figsize=(40,30))
	plt.plot(polynomial_degrees,train_errors_mse,"g-", label="ln MSE for training dataset")
	plt.plot(polynomial_degrees,test_errors_mse,"r-", label="ln MSE for test dataset")

	plt.xlabel("Degree of Polynomial", fontsize=14) # Increased font size for labels
	plt.ylabel("MSE", fontsize=14)

	# Get the min and max values of X and y
	pd_min, pd_max = 0, 30
	em_min, em_max = -15, 40

	# It's good practice to add a little padding to the axis limits
	pd_padding = (pd_max - pd_min) * 0.05
	em_padding = (em_max - em_min) * 0.05

	# Set the axis limits with padding
	plt.axis([pd_min - pd_padding, pd_max + pd_padding, em_min - em_padding, em_max + em_padding])

	plt.grid(True) # Ensure grid is visible
	plt.legend(loc="upper left", fontsize=12) # Increased font size for legend

	plt.title("ln MSE for Training and test dataset", fontsize=16)
	plt.show()


np.random.seed(1836)  # to make this code example reproducible
m = 300  # number of instances
X = 3 * np.random.rand(m, 1)  # column vector for independent variable
y = 3 + np.random.uniform(0, 10,1) * X + np.random.exponential(10,1) * X**2 - np.random.uniform(0,10,1) * X**3 +np.random.uniform(0,30,1)*X**4+ np.random.normal(0,10,1) * np.random.randn(m, 1) # dependent variable with cubic relationship and noise

# --- Split Data into Training and Testing Sets ---
# We'll use 80% of the data for training and 20% for testing.
# random_state ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1836)


theta_best_linear=parameter_linear_regression(X_train,y_train)
theta_best_cubic=parameter_cubic_regression(X_train,y_train)

polynomial_degrees=range(1,15)
train_errors_mse = []
test_errors_mse = []
train_errors_mse = []
test_errors_mse = []

for degree in polynomial_degrees:
    deg_train = X_train
    deg_test = X_test
    theta_best_polly = parameter_polynomial_regression(X_train, y_train, degree)
    X_train_poly_b_new=polly_train_X(X_train,degree)
    X_test_poly_b_new=polly_train_X(X_test,degree)

    # Assuming you create polynomial features and dummy feature inside your regression function or separately
    # If not, you need to create X_train_poly_b_new and X_test_poly_b_new here according to degree

    y_train_acc = y_train
    y_test_acc = y_test
    y_train_pred = X_train_poly_b_new @ theta_best_polly
    y_test_pred = X_test_poly_b_new @ theta_best_polly

    train_errors_mse.append(mean_squared_error(y_train_pred, y_train_acc))
    test_errors_mse.append(mean_squared_error(y_test_pred, y_test_acc))


#print(train_errors_mse)
#print(test_errors_mse)
#comperison_linear_and_cubic(X,y,theta_best_linear,theta_best_cubic)
mse_in_various_deg (polynomial_degrees,train_errors_mse,test_errors_mse)
#mse_in_various_deg (polynomial_degrees,np.log(train_errors_mse),np.log(test_errors_mse))


		





