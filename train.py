"""
Train the Heart Disease Neural Network.
"""

import matplotlib.pyplot as plt

from data.dataset import load_dataset

from nn.model import NeuralNetwork

from nn.utils import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_dataset()

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print()

    print("=" * 60)
    print("Building Neural Network...")
    print("=" * 60)

    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size1=16,
        hidden_size2=8,
        learning_rate=0.01
    )

    print()

    print("=" * 60)
    print("Training...")
    print("=" * 60)

    model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32
)

    model.save(
        "outputs/model_weights.npz"
    )

    print("Model saved successfully!")

    print()

    print("=" * 60)
    print("Evaluating...")
    print("=" * 60)

    train_predictions = model.predict(X_train)

    test_predictions = model.predict(X_test)

    print()

    print("Training Accuracy :",
          accuracy_score(y_train, train_predictions))

    print("Test Accuracy     :",
          accuracy_score(y_test, test_predictions))

    print("Precision         :",
          precision_score(y_test, test_predictions))

    print("Recall            :",
          recall_score(y_test, test_predictions))

    print("F1 Score          :",
          f1_score(y_test, test_predictions))

    print()

    print("Confusion Matrix")

    print(confusion_matrix(
        y_test,
        test_predictions
    ))

    print()

    plt.figure(figsize=(8,5))

    plt.plot(
        model.loss_history,
        linewidth=2
    )

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Binary Cross Entropy Loss")

    plt.grid(True)

    plt.savefig("outputs/loss_curve.png")

    plt.show()


if __name__ == "__main__":
    main()