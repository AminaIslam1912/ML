import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def load_data(filepath):
    df = pd.read_csv(filepath)

    X = df.iloc[:, 0].values   # feature
    y = df.iloc[:, 1].values   # target

    return X, y


# ==============================
# 2. Plot Feature
# ==============================

def plot_feature(X, y):
    plt.figure()
    plt.scatter(X, y)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Feature vs Target")
    plt.show()


# ==============================
# 3. Train/Validation Split
# ==============================

def split_data(X, y):

    np.random.seed(42)
    indices = np.random.permutation(len(X))

    split = int(0.8 * len(X))

    train_idx = indices[:split]
    val_idx = indices[split:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# ==============================
# 4. Create Polynomial Features
# ==============================

def create_polynomial_features(X, degree):

    X_poly = []

    for x in X:
        row = []
        for d in range(1, degree + 1):
            row.append(x ** d)
        X_poly.append(row)

    return np.array(X_poly)


# ==============================
# 4b. Normalize Features
# ==============================

def normalize(X_train, X_val):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, mean, std




class LinearRegression:

    def __init__(self, lr=None, epochs=5000):
        self.lr = lr
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
        safe_lr = n / (2 * lambda_max) if lambda_max > 0 else 0.01

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
                print(f"  Stopped early at epoch {i} due to numerical instability.")
                break

            self.train_errors.append(train_mse)
            self.val_errors.append(val_mse)

    def best_validation_error(self):
        best_epoch = np.argmin(self.val_errors)
        return self.val_errors[best_epoch], best_epoch




if __name__ == "__main__":

    X, y = load_data("data_02b.csv")

    
    plot_feature(X, y)

    X_train, X_val, y_train, y_val = split_data(X, y)

    degrees = [1, 2, 3]

    val_errors = []
    models = []
    norm_params = []   
    X_plot = np.linspace(min(X), max(X), 200)

    for d in degrees:

        
        X_train_poly = create_polynomial_features(X_train, d)
        X_val_poly = create_polynomial_features(X_val, d)

    
        X_train_norm, X_val_norm, mean, std = normalize(X_train_poly, X_val_poly)
        norm_params.append((mean, std))

        model = LinearRegression(lr=0.01, epochs=5000)
        model.train(X_train_norm, y_train, X_val_norm, y_val)

        best_val, best_epoch = model.best_validation_error()

        val_errors.append(best_val)
        models.append(model)

        print(f"\nDegree {d}")
        print("Best Epoch:", best_epoch)
        print("Best Validation MSE:", best_val)
        print("Learned Parameters:", model.theta)

    
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Data", alpha=0.5)

    for i, d in enumerate(degrees):
        X_plot_poly = create_polynomial_features(X_plot, d)
        
        mean, std = norm_params[i]
        X_plot_norm = (X_plot_poly - mean) / std
        X_plot_bias = models[i].add_bias(X_plot_norm)
        y_plot = X_plot_bias.dot(models[i].theta)
        plt.plot(X_plot, y_plot, label=f"d = {d}")

    plt.title("Polynomial Regression Fits (d = 1, 2, 3)")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.show()

   
    for i, d in enumerate(degrees):
        plt.figure(figsize=(8, 5))
        plt.plot(models[i].train_errors, label="Training Error")
        plt.plot(models[i].val_errors, label="Validation Error")
        plt.title(f"Error Curve (d = {d})")
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.show()

   
    plt.figure(figsize=(6, 4))
    plt.bar(["d=1", "d=2", "d=3"], val_errors)
    plt.title("Validation Errors for Different Degrees")
    plt.ylabel("Validation MSE")
    plt.tight_layout()
    plt.show()

   
    best_index = np.argmin(val_errors)
    best_degree = degrees[best_index]
    best_model = models[best_index]

    print("\n====================================")
    print("BEST DEGREE:", best_degree)
    print("Best Validation MSE:", val_errors[best_index])
    print("Best Parameters (theta):", best_model.theta)
    print("====================================")