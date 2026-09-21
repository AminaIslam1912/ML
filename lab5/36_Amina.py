import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def load_diabetes_data(csv_path="diabetes.csv"):
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


def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def misclassification_error(y_true, y_pred_probs, threshold=0.5):
    y_pred_labels = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_pred_labels != y_true)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def perceptron_train(X, y, X_val, y_val, lr=0.01, n_iters=100):
    Xb = add_bias(X)
    Xb_val = add_bias(X_val)
    weights = np.zeros(Xb.shape[1])
    train_misclass = []
    val_misclass = []

    for _ in range(n_iters):
        y_pred_train = (Xb @ weights >= 0).astype(int)
        y_pred_val = (Xb_val @ weights >= 0).astype(int)
        train_misclass.append(np.mean(y_pred_train != y))
        val_misclass.append(np.mean(y_pred_val != y_val))
        errors = y - y_pred_train
        # Single vectorized weight update per epoch
        weights += lr * (Xb.T @ errors)

    return weights, train_misclass, val_misclass


def perceptron_predict(X, weights):
    Xb = add_bias(X)
    return (Xb @ weights >= 0).astype(int)


def logistic_regression_train(X, y, X_val, y_val, lr=0.1, n_iters=200):
    Xb = add_bias(X)
    Xb_val = add_bias(X_val)
    weights = np.zeros(Xb.shape[1])
    train_misclass = []
    val_misclass = []
    train_log_losses = []
    val_log_losses = []

    for _ in range(n_iters):
        probs_train = sigmoid(Xb @ weights)
        probs_val = sigmoid(Xb_val @ weights)
        gradient = (Xb.T @ (probs_train - y)) / Xb.shape[0]
        weights -= lr * gradient
        train_misclass.append(misclassification_error(y, probs_train))
        val_misclass.append(misclassification_error(y_val, probs_val))
        train_log_losses.append(log_loss(y, probs_train))
        val_log_losses.append(log_loss(y_val, probs_val))

    return weights, train_misclass, val_misclass, train_log_losses, val_log_losses


def logistic_regression_predict_proba(X, weights):
    Xb = add_bias(X)
    return sigmoid(Xb @ weights)


def evaluate_model(y_true, y_pred_labels, model_name):
    cm = confusion_matrix(y_true, y_pred_labels)
    acc = accuracy_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels)
    rec = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    print(f"\n{model_name} Metrics")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")


def plot_curves(train_values, val_values, title, ylabel, filename, show=True):
    plt.figure()
    plt.plot(range(1, len(train_values) + 1), train_values, label="Train")
    plt.plot(range(1, len(val_values) + 1), val_values, label="Validation")
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
    data = load_diabetes_data()
    X = data.drop(columns=["Outcome"]).values
    y = data["Outcome"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Perceptron
    p_weights, p_train_misclass, p_val_misclass = perceptron_train(
        X_train, y_train, X_val, y_val, lr=0.01, n_iters=100
    )
    p_val_pred = perceptron_predict(X_val, p_weights)
    evaluate_model(y_val, p_val_pred, "Perceptron")

    # Logistic Regression
    (
        lr_weights,
        lr_train_misclass,
        lr_val_misclass,
        lr_train_losses,
        lr_val_losses,
    ) = logistic_regression_train(
        X_train, y_train, X_val, y_val, lr=0.1, n_iters=200
    )
    lr_val_probs = logistic_regression_predict_proba(X_val, lr_weights)
    lr_val_pred = (lr_val_probs >= 0.5).astype(int)
    evaluate_model(y_val, lr_val_pred, "Logistic Regression")

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_val_probs = nb_model.predict_proba(X_val)[:, 1]
    nb_val_pred = (nb_val_probs >= 0.5).astype(int)
    evaluate_model(y_val, nb_val_pred, "Naive Bayes (GaussianNB)")

    # Error curves
    plot_curves(
        p_train_misclass,
        p_val_misclass,
        "Perceptron Misclassification Rate",
        "Misclassification Rate",
        "perceptron_misclassification.png",
        show=True,
    )
    plot_curves(
        lr_train_misclass,
        lr_val_misclass,
        "Logistic Regression Misclassification Rate",
        "Misclassification Rate",
        "logreg_misclassification.png",
        show=True,
    )
    plot_curves(
        lr_train_losses,
        lr_val_losses,
        "Logistic Regression Log Loss",
        "Log Loss",
        "logreg_logloss.png",
        show=True,
    )


if __name__ == "__main__":
    main()
