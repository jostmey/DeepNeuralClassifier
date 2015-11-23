## Description

No non-sense example of a deep neural network in Julia written from scratch. No machine learning packages are used.

The neural network is trained to reconize hand written characters from the MNIST dataset. The neural network is four layers deep, made up of a sigmoid layer (data layer), followed by three softplus layers, and ending with a softmax layer. No geometric invariances are assumed-this is not a convolutional neural network. Dropout is used to prevent overfitting.

Normally the training data should be split into a training set and a validation set so that the hyperparameters such as the learning rate can be optimized. However, non of the hyperparameters were adjusted, so a validation set was not needed.

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

Backpropagation
ReLu
Dropout
Lack of validation set... no hyperparameter optimization

