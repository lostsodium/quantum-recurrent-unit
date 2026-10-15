# cnn_param_count — Parameter count of the CNN benchmark (Table 7)

Counts the trainable parameters of the CNN benchmark of Bowles et al. [22] for the 8×8 MNIST "3" vs "5" dataset,
using their released model with its reported best hyperparameters for this dataset. Result: **51,841** trainable
parameters (Table 7, Table 11).

| File | Description |
|---|---|
| `count_cnn_params.ipynb` | Instantiates the CNN and counts its parameters layer by layer |
| `convolutional_neural_network.py` | CNN model from the QML benchmarks repository of Bowles et al. (unmodified) |
| `ConvolutionalNeuralNetwork_mnist_pixels_3-5_8x8_GridSearchCV-best-hyperparams.csv` | Best hyperparameters for the 8×8 dataset, from the same repository |

The convolutional output channels are not part of the grid search; the class default `[32, 64]` is used.
Training is not needed to count parameters, so `qml_benchmarks.model_utils` is replaced by an empty stub in the
notebook.

## Source and license

`convolutional_neural_network.py` and the hyperparameter file are taken from
https://github.com/XanaduAI/qml-benchmarks (Copyright 2024 Xanadu Quantum Technologies Inc.), licensed under
the Apache License, Version 2.0. The license header is retained in the file.

Reference: J. Bowles, S. Ahmed, and M. Schuld, "Better than classical? The subtle art of benchmarking quantum
machine learning models," arXiv:2403.07059 (2024).
