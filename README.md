## Description

See example scripts of a deep neural network coded from scratch. No machine learning packages are used, exposing the underlying algorithms. The code is written in the Julia, a programming language with a syntax similar to Matlab.

In this example, the neural network is trained on the MNIST dataset of hand written digits. On the test dataset, the neural network correctly classifies XX % of the hand written digits. These are near state of the art results for a neural network that does not contain any information about the geometric invariances of the data.

## Download

* Download: [zip](https://github.com/jostmey/DeepNeuralClassifieer/zipball/master)
* Git: `git clone https://github.com/jostmey/DeepNeuralClassifier`

## REQUIREMENTS

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). Make sure the version of Julia is no older than v0.4.

## RUN

The scripts will not work without first adding the MNIST dataset. Launch Julia, which can be done by opening the command line terminal and typing `julia`. At the prompt, run `Pkg.add("MNIST")`.

Once the MNIST dataset has been added the neural network can be trained. The procedure can take several days. Set the working directory to this folder and run the following in the command line terminal.

`julia train.jl > train.out`

The neural network will save its parameters to a folder called `bin/` once training is complete. To use the neural network to classify the hand written digits in the test set, run the following command.

`julia test.jl > test.out`

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

