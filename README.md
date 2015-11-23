## Description

Example scripts of a deep neural network coded from scratch. No machine learning packages are used, exposing the underlying algorithms. The code is written in the Julia, a programming language with a syntax similar to matlab.

In this example, the neural network is trained on the MNIST dataset of hand written digits. On the test dataset, the neural network correctly classifies XX % of the hand written digits. These are near state of the art results for a neural network that does not include a priori any geometric invariances about the dataset.

## Download

* Download: [zip](https://github.com/jostmey/DeepNeuralClassifieer/zipball/master)
* Git: `git clone https://github.com/jostmey/DeepNeuralClassifier`

## REQUIREMENTS

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). Make sure the version is at lest 0.4.

Pkg.add("MNIST")

## RUN

julia train.jl > train.out

julia test.jl > test.out

## THEORY

* Backpropagation using Cross-entropy error function (Likelihood optimization routine)
* ReLu (No need to pretrain)
* Dropout
* Hyperparameters (linear decay in learning rate, no hyperparameter optimization)

| Layer | Neuron Type | Purpose                  |
| :----:|:-----------:|:-------------------------|
| 1     | Sigmoid     | Normalize Features       |
| 2     | Softplus    | Nonlinear Transformation |
| 3     | Softplus    | Nonlinear Transformation |
| 4     | Softplus    | Nonlinear Transformation |
| 5     | Softmax     | Decision Layer           |

