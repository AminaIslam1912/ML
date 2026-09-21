import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def load_data(csv_path="diabetes.csv"):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    if not os.path.exists(csv_path):
        data = pd.read_csv(url, header=None, names=columns)
        data.to_csv(csv_path, index=False)
    data = pd.read_csv(csv_path)
    print(data.head())
    return data


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def misclass_error(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_pred != y_true)


def train_perceptron(X, y, X_val, y_val, lr=0.01, n_iters=100):
    Xb = add_bias(X)
    Xb_val = add_bias(X_val)
    w = np.zeros(Xb.shape[1])
    train_err = []
    val_err = []

    for _ in range(n_iters):
        pred_train = (Xb @ w >= 0).astype(int)
        pred_val = (Xb_val @ w >= 0).astype(int)
        train_err.append(np.mean(pred_train != y))
        val_err.append(np.mean(pred_val != y_val))
        w += lr * (Xb.T @ (y - pred_train))

    return w, train_err, val_err


def train_logistic(X, y, X_val, y_val, lr=0.1, n_iters=200):
    Xb = add_bias(X)
    Xb_val = add_bias(X_val)
    w = np.zeros(Xb.shape[1])
    train_err = []
    val_err = []
    train_loss = []
    val_loss = []

    for _ in range(n_iters):
        p_train = sigmoid(Xb @ w)
        p_val = sigmoid(Xb_val @ w)
        grad = (Xb.T @ (p_train - y)) / Xb.shape[0]
        w -= lr * grad
        train_err.append(misclass_error(y, p_train))
        val_err.append(misclass_error(y_val, p_val))
        train_loss.append(log_loss(y, p_train))
        val_loss.append(log_loss(y_val, p_val))

    return w, train_err, val_err, train_loss, val_loss


def evaluate(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name} Metrics")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")


def plot_two_curves(train_vals, val_vals, title, ylabel, filename, show=True):
    plt.figure()
    plt.plot(range(1, len(train_vals) + 1), train_vals, label="Train")
    plt.plot(range(1, len(val_vals) + 1), val_vals, label="Validation")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    data = load_data()
    X = data.drop(columns=["Outcome"]).values
    y = data["Outcome"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Perceptron
    w_p, p_train_err, p_val_err = train_perceptron(X_train, y_train, X_val, y_val)
    p_val_pred = (add_bias(X_val) @ w_p >= 0).astype(int)
    evaluate(y_val, p_val_pred, "Perceptron")

    # Logistic Regression
    w_lr, lr_train_err, lr_val_err, lr_train_loss, lr_val_loss = train_logistic(
        X_train, y_train, X_val, y_val
    )
    lr_val_probs = sigmoid(add_bias(X_val) @ w_lr)
    lr_val_pred = (lr_val_probs >= 0.5).astype(int)
    evaluate(y_val, lr_val_pred, "Logistic Regression")

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_val_pred = (nb.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
    evaluate(y_val, nb_val_pred, "Naive Bayes (GaussianNB)")

    # Curves
    plot_two_curves(
        p_train_err,
        p_val_err,
        "Perceptron Misclassification Rate",
        "Misclassification Rate",
        "perceptron_misclassification.png",
        show=True,
    )
    plot_two_curves(
        lr_train_err,
        lr_val_err,
        "Logistic Regression Misclassification Rate",
        "Misclassification Rate",
        "logreg_misclassification.png",
        show=True,
    )
    plot_two_curves(
        lr_train_loss,
        lr_val_loss,
        "Logistic Regression Log Loss",
        "Log Loss",
        "logreg_logloss.png",
        show=True,
    )


if __name__ == "__main__":
    main()
