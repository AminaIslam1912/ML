
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def load_data(filepath):
    df = pd.read_excel(filepath)

    
    X = df.iloc[:, 0:4].values

   
    y = df.iloc[:, 4].values

    return X, y



def plot_features(X, y):
    feature_names = ["Temperature (T)", "Exhaust Vacuum (V)",
                     "Ambient Pressure (AP)", "Relative Humidity (RH)"]

    for i in range(4):
        plt.figure()
        plt.scatter(X[:, i], y)
        plt.xlabel(feature_names[i])
        plt.ylabel("Electrical Power Output (EP)")
        plt.title(feature_names[i] + " vs EP")
        plt.show()



def split_data(X, y):

    np.random.seed(42)
    indices = np.random.permutation(len(X))

    split = int(0.8 * len(X))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, X_val, y_train, y_val


def normalize(X_train, X_val):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val



class LinearRegression:

    def __init__(self, learning_rate=None, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.effective_lr = None

    def add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def train(self, X_train, y_train, X_val, y_val):

        X_train = self.add_bias(X_train)
        X_val = self.add_bias(X_val)

        self.theta = np.zeros(X_train.shape[1])

        self.train_errors = []
        self.val_errors = []

        n = len(y_train)

        
        gram = X_train.T.dot(X_train)
        lambda_max = np.linalg.eigvalsh(gram).max()
        if lambda_max > 0:
            safe_lr = n / (2 * lambda_max)
        else:
            safe_lr = 0.01

        if self.lr is None:
            self.effective_lr = safe_lr
        else:
            self.effective_lr = min(self.lr, safe_lr)

        for i in range(self.epochs):

            predictions = X_train.dot(self.theta)

            error = predictions - y_train

            gradient = (2/n) * X_train.T.dot(error)

            self.theta = self.theta - self.effective_lr * gradient

            
            train_mse = np.mean((X_train.dot(self.theta) - y_train) ** 2)
            val_mse = np.mean((X_val.dot(self.theta) - y_val) ** 2)

            if (not np.all(np.isfinite(self.theta)) or
                    not np.isfinite(train_mse) or
                    not np.isfinite(val_mse)):
                print("Stopped early due to numerical instability.")
                break

            self.train_errors.append(train_mse)
            self.val_errors.append(val_mse)

    def plot_learning_curve(self):

        plt.figure()
        plt.plot(self.train_errors, label="Training Error")
        plt.plot(self.val_errors, label="Validation Error")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()

    def best_validation_error(self):

        best_epoch = np.argmin(self.val_errors)

        return (self.val_errors[best_epoch],
                self.train_errors[best_epoch],
                best_epoch)




def run(X, y, use_normalization=False):

    X_train, X_val, y_train, y_val = split_data(X, y)

    if use_normalization:
        X_train, X_val = normalize(X_train, X_val)

   
    model = LinearRegression(
        learning_rate=0.001 if use_normalization else None,
        epochs=10000
    )

    model.train(X_train, y_train, X_val, y_val)

    model.plot_learning_curve()

    best_val, best_train, epoch = model.best_validation_error()

    print("====================================")
    print("Normalization Used:", use_normalization)
    print("Best Epoch:", epoch)
    print("Best Validation MSE:", best_val)
    print("Training MSE at Best Epoch:", best_train)
    print("Effective Learning Rate:", model.effective_lr)
    print("Learned Parameters (theta):")
    print(model.theta)
    print("====================================\n")




if __name__ == "__main__":

    X, y = load_data("Folds5x2_pp.xlsx")

    plot_features(X, y)

    print("WITHOUT NORMALIZATION")
    run(X, y, use_normalization=False)

    print("WITH NORMALIZATION")
    run(X, y, use_normalization=True)