"""
Predict Heart Disease using saved Neural Network model.
"""

from data.dataset import load_dataset
from nn.model import NeuralNetwork


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_dataset()


    print()
    print("=" * 60)
    print("Loading trained model...")
    print("=" * 60)

    model = NeuralNetwork(
        input_size=X_test.shape[1],
        hidden_size1=16,
        hidden_size2=8
    )

    model.load(
        "outputs/model_weights.npz"
    )

    print("Model loaded successfully!")


    print()
    print("=" * 60)
    print("Heart Disease Predictions")
    print("=" * 60)


    probabilities = model.predict_proba(
        X_test
    )

    predictions = model.predict(
        X_test
    )


    for i in range(10):

        probability = probabilities[i][0]
        prediction = predictions[i][0]
        actual = y_test[i][0]


        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"


        print()
        print(f"Sample {i+1}")
        print("-" * 30)
        print(f"Actual Label   : {actual}")
        print(f"Probability    : {probability:.4f}")
        print(f"Prediction     : {prediction}")
        print(f"Result         : {result}")


if __name__ == "__main__":
    main()