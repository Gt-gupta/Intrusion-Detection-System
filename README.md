## Overview

This project implements a custom Artificial Neural Network (ANN) from scratch in Python to classify network connections as Malicious or Benign.
It uses pandas and scikit-learn for preprocessing and data handling, and numpy for the ANN implementation.

The goal is to demonstrate how a simple ANN can be built without deep learning frameworks, while still handling real-world network traffic data.
## Features

Data preprocessing: 
Drops irrelevant network traffic columns.
Converts categorical features (conn_state, history) into one-hot encoded vectors.
Handles missing / placeholder values ('-' replaced with 0).
Normalizes features with StandardScaler.

Neural Network (from scratch):
One hidden layer with customizable size.
Sigmoid activation functions.
Forward & backward propagation.
Gradient descent optimization.
Training/validation split with accuracy evaluation.
## Dataset

The dataset (data.csv) contains network connection logs with features like:

Numerical: duration, bytes, packets, etc.

Categorical: connection state, history.

Label: Benign or Malicious.

Columns automatically dropped include technical fields (timestamps, IPs, tunnel parents, etc.).

## Future Improvements

Add more layers and activation functions.

Implement batch training and early stopping.

Replace manual ANN with PyTorch/TensorFlow for larger datasets.

Evaluate using metrics beyond accuracy (precision, recall, F1-score).

