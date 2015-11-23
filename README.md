## Description

See example scripts of a deep, feed-forward neural network coded from scratch. No machine learning packages are used, exposing the underlying algorithms. The code is written in the Julia, a programming language with a syntax similar to Matlab.

In this example, the neural network is trained on the MNIST dataset of hand written digits. On the test dataset, the neural network correctly classifies XX % of the hand written digits. These are near state of the art results for a neural network that does not contain any information about the geometric invariances of the data.

## Download

* Download: [zip](https://github.com/jostmey/DeepNeuralClassifieer/zipball/master)
* Git: `git clone https://github.com/jostmey/DeepNeuralClassifier`

## REQUIREMENTS

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). The code has been tested on version 0.4, and will not run on earlier versions.

## RUN

The scripts will not work without first adding the MNIST dataset. Launch `julia` and at at the prompt run `Pkg.add("MNIST")`.

Training the neural network can take several days. Set the working directory to this folder and run the following in the command line terminal.

`julia train.jl > train.out`

The neural network will save its parameters to a folder called `bin/` once training is complete. To classify the all the hand written digits in the test set, run the following command.

`julia test.jl > test.out`

## THEORY

Feed-forward neural networks are commonly trained using backpropagation to optimize some objective function. The backpropagation algorithm is an efficient way of computing the gradient of a neural network by passing the error at the output layer backward through each layer. Each backward pass amounts to applying the chain-rule from Calculus on the objective function. In this example, the cross-entropy error is used as the objective function. Using the cross-entropy as the error function is an ideal choice because it is equivalient to optimizing the likelihood function. After computing the errors from backprogation at each layer for several cases, a small change in the weights and biases are made. The collection of changes is called a minibatch.

Deep neural networks made of sigmoidal neurons suffer from the vanishing gradient problem. This is where the errors in the backpropagation pass become smaller and smaller each after each layer. By the time the top layer is reached the errors are almost zero. An intractable number of updates would be required to train the neural network. To overcome the shortcoming of the sigmoidal units, Rectified Linear units were introduced. Here, a smooth generalization of the Rectified Linear unit is used called the Softmax unit. Becaus the output is not a binary response, a softmax unit is used in the last layer of the neural network to choose from the list of handwritten digits. The neural architecture is summarizer in the following table.

| Layer | Neuron Type | Purpose                  | Number |
| :----:|:-----------:|:-------------------------|:------:|
| 1     | Sigmoid     | Normalize Features       | 28^2   |
| 2     | Softplus    | Nonlinear Transformation | 2000   |
| 3     | Softplus    | Nonlinear Transformation | 2000   |
| 4     | Softplus    | Nonlinear Transformation | 2000   |
| 5     | Softmax     | Decision Layer           | 1      |

* Hyperparameters
* Regularization


