... WORK IN PROGESS ...SUBJECT TO HEAVY REVISION

## Description

Example scripts for a deep, feed-forward neural network have been coded from scratch. No machine learning packages are used, leaving the underling alrogithms exposed and viewable. The code is written in the Julia, a programming language with a syntax similar to Matlab.

The neural network is trained on the MNIST dataset of hand written digits. On the test dataset, the neural network correctly classifies XX % of the hand written digits. The results are state of the art for a neural network that does not contain a priori knowledge about the geometric invariances of the dataset like a Convolutional Neural Network does.

## Download

* Download: [zip](https://github.com/jostmey/DeepNeuralClassifieer/zipball/master)
* Git: `git clone https://github.com/jostmey/DeepNeuralClassifier`

## REQUIREMENTS

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). The scripts have been developed using version 0.4 and do not work on previous versions of Julia.

## RUN

You must first add the MNIST dataset to Julia. Launch `julia` and run `Pkg.add("MNIST")` at the prompt.

Training the neural network can take several days. Set the working directory to this folder and run the following in the command line terminal.

`julia train.jl > train.out`

The neural network will save its parameters to a folder called `bin/` once training is complete. To classify the hand written digits in the test set, run the following command.

`julia test.jl > test.out`

The percentage of correct answers will be written at the end of the text file `test.out`.

## THEORY

###### Training

Feed-forward neural networks are commonly trained using Backpropagation to minimize the error between the desired and actual output. The Backpropgation algorithm is an efficient method for computing the gradient of the error. The error from the output is passed backward through the weights of the neural network, multipling the errors by the derivative of that layer. Each pass of the errors through a previous layer amounts to carrying out the Chain rule from calculus to compute the derivative. The error-loss is minimized by moving the weights of the neural network down the gradient. Changes are made to the weights in small, discrete steps determined by the value put in the *learning rate*.

Minimizing the cross-entropy error is equivalent to maximizing the Likelihood function, allowing us to train neural networks using Maximum Likelihood methods. To follow the true gradient of the Likelihood function would require using the entire dataset to update the weights each iteration. In practice, only a minibatch is used at each iteration with unused examples set aside to use in future minibatches. That said, theoretical evidence exists indicating that the use of minibatches will not distort the objective function if the learning rate is decreased at each iteration following a specific schedule. The schedule used here is an approximation---the learning rate is decreased following a linear progression.

% A momentum term is included to help the training procedure escape inflection points and local minima. The idea is that adding momentum will help the parameter overcome

% A momentum term is included to help the training procedure escape local traps. The idea is that the a momentum term will keep the training procedure moving over inflection points. Normally a series of simulations would be run to find the optimal hyperparameters. This would require splitting the training data into a training set and a validation set. However, the results reported here are from only one training run. Therefore, no validation set is required.

###### Architecture

Deep neural networks made of sigmoidal neurons suffer from the vanishing gradient problem. This is where the errors in the backpropagation pass become smaller and smaller each after each layer. By the time the top layer is reached the errors are almost zero. An intractable number of updates would be required to train the neural network. To overcome the shortcoming of the sigmoidal units, Rectified Linear units were introduced. Rectified linear units are linear over all inputs greater than zero, and hence the derivative is well behaved over this region. The units still introduce a nonlinearty by defining boundaries where values are zero. Here, a smooth generalization of the Rectified Linear unit is used called the Softmax unit. Because the output is not a binary response, a softmax unit is used in the last layer of the neural network to choose from the list of handwritten digits. The neural architecture is summarizer in the following table.

| Layer | Neuron Type | Purpose                  | Number |
| :----:|:-----------:|:-------------------------|:------:|
| 1     | Sigmoid     | Normalize Features       | 28^2   |
| 2     | Softplus    | Nonlinear Transformation | 500    |
| 3     | Softplus    | Nonlinear Transformation | 500    |
| 4     | Softplus    | Nonlinear Transformation | 500    |
| 5     | Softmax     | Decision Layer           | 1      |

###### Regularization



