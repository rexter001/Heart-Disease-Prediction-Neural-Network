# Heart Disease Prediction using Neural Network From Scratch

A Machine Learning project that predicts the presence of heart disease using a **Neural Network implemented completely from scratch using NumPy**.

This project focuses on understanding the internal working of neural networks by implementing the core concepts manually instead of using high-level deep learning frameworks like TensorFlow or PyTorch.

---

## Project Overview

Heart disease is one of the leading causes of mortality worldwide. This project uses patient medical attributes to build a binary classification model that predicts whether a person is likely to have heart disease.

The neural network performs:

* Forward propagation
* Loss calculation
* Backpropagation
* Weight updates using gradient descent
* Binary classification prediction

All neural network components are implemented manually.

---

## Key Features

* Neural Network built from scratch using NumPy
* Custom layers and activation functions
* Manual forward propagation
* Manual backpropagation
* Binary classification using sigmoid activation
* Model weight saving and loading
* Prediction probability output
* Loss curve visualization

---

## Model Architecture

The implemented neural network contains:

```
Input Layer
     |
     ↓
Dense Layer (16 neurons)
     |
     ↓
ReLU Activation
     |
     ↓
Dense Layer (8 neurons)
     |
     ↓
ReLU Activation
     |
     ↓
Output Layer (1 neuron)
     |
     ↓
Sigmoid Activation
     |
     ↓
Binary Prediction
```

---

## Dataset

The project uses the Heart Disease dataset containing medical information of patients.

Features include:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Maximum heart rate
* Exercise-induced angina
* Other clinical attributes

Target:

```
0 → No Heart Disease
1 → Heart Disease Present
```

Dataset location:

```
data/heart.csv
```

---

## Project Structure

```
Heart Disease Prediction
│
├── data
│   ├── heart.csv
│   └── dataset.py
│
├── nn
│   ├── model.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   └── utils.py
│
├── outputs
│   └── loss_curve.png
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python train.py
```

The training process will:

* Load and preprocess the dataset
* Initialize the neural network
* Train the model
* Display training loss
* Save trained weights

---

## Making Predictions

Run:

```bash
python predict.py
```

The model outputs:

* Actual label
* Prediction probability
* Predicted class
* Final result

Example:

```
Sample 1
------------------------------
Actual Label   : 0
Probability    : 0.7421
Prediction     : 1
Result         : Heart Disease Detected
```

---

## Model Performance

Evaluation on test data:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 85.33% |
| Precision | 80.95% |
| Recall    | 92.39% |
| F1 Score  | 86.29% |

Confusion Matrix:

```
[[72 20]
 [ 7 85]]
```

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Git

---

## Why Build From Scratch?

Instead of using pre-built machine learning libraries, this project implements the fundamental components of a neural network manually.

This provides a deeper understanding of:

* How neurons calculate outputs
* How errors propagate backward
* How weights are optimized
* How learning happens inside neural networks

---

## Future Improvements

* Add a web-based user interface
* Deploy the model as an API
* Add real-time patient input prediction
* Improve model visualization
* Experiment with different architectures

---

## Author

- [@Khaja Mastan](https://www.github.com/rexter001)

B.Tech Computer Science Engineering Student

---

## License

This project is licensed under the MIT License.
