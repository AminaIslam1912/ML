import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def load_data():
    data = pd.read_csv("data_01.csv", header=None, names=["x", "y"])
    return data



def process_data(data):
    x = data["x"].values.reshape(-1, 1)
    y = data["y"].values.reshape(-1, 1)

    
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_scaled = (x - x_mean) / x_std
    
    X = np.hstack([np.ones((len(x), 1)), x_scaled])
    # X = np.hstack([np.ones((len(x), 1)), x])

    return X, y, x_mean, x_std



def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error**2)
    return cost



def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X @ theta
        gradient = (1/m) * (X.T @ (predictions - y))
        theta = theta - alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


def train(X, y):
    theta = np.zeros((2,1))
    alpha = 0.01  
    iterations = 1000

    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    return theta, cost_history



def evaluate(X, y, theta):
    print("theta0 =", theta[0][0])
    print("theta1 =", theta[1][0])
    print("Final cost =", compute_cost(X, y, theta))



def main():
    data = load_data()

    
    plt.scatter(data["x"], data["y"])
    plt.title("Dataset from CSV")
    plt.show()

    X, y, x_mean, x_std = process_data(data)

    theta, cost_history = train(X, y)

    evaluate(X, y, theta)

    
    plt.plot(cost_history)
    plt.title("Training Error")
    plt.show()

  
    
    predictions = X @ theta
    plt.scatter(data["x"], data["y"])
    plt.plot(data["x"], predictions, color="red")
    plt.title("Regression Line")
    plt.show()


main()