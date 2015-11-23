## Description

A simple example of a deep neural network written in Julia is deposited here. The scripts are written from scratch and do not rely on existing Machine Learning packages.

In this example, the neural network is trained to recognize hand written characters from the MNIST dataset. The five layers listed from top to bottom contain the following neuron types.

| Layer | Neuron Type | Purpose                  |
| :----:|:-----------:|:------------------------:|
| 1     | Sigmoid     | Normalize Features       |
| 2     | Softplus    | Nonlinear Transformation |
| 3     | Softplus    | Nonlinear Transformation |
| 4     | Softplus    | Nonlinear Transformation |
| 5     | Softmax     | Decision Layer           |

No geometric invariances are assumed -- this is not a convolutional neural network. 

## Download

* Download: [zip](https://github.com/jostmey/DeepNeuralClassifieer/zipball/master)
* Git: `git clone https://github.com/jostmey/DeepNeuralClassifier`

## REQUIREMENTS

Julia v4.0

Pkg.add("MNIST")

## RUN

julia train.jl > train.out

julia test.jl > test.out

## THEORY

* Backpropagation using Cross-entropy error function (Likelihood optimization routine)
* ReLu (No need to pretrain)
* Dropout
* Hyperparameters (linear decay in learning rate, no hyperparameter optimization)

