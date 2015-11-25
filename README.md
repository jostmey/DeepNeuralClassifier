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

Feed-forward neural networks are commonly trained using Backpropagation to minimize the error between the desired and actual output. The Backpropgation algorithm is an efficient method for computing the gradient of the error-loss function. The error from the output is passed backward through the weights of the neural network, multipling the values of the backpropagated errors by the derivative of that layer. Each backward pass through a previous layer amounts to carrying out the Chain rule (from Calculus) to compute the derivative. The error-loss is minimized by moving the weights of the neural network down the gradient. Changes are made to the weights in small, discrete steps determined by the value of the *learning rate*.

Minimizing the cross-entropy error is equivalent to maximizing the Likelihood function, allowing us to train neural networks using Maximum Likelihood methods. To follow the true gradient of the Likelihood function would require using the entire dataset to update the weights each iteration. In practice, only a minibatch is used at each iteration. THe unused examples are set aside to use in future minibatches. That said, theoretical evidence suggests that the use of minibatches will not distort the objective function if the learning rate is decreased at each iteration following a specific schedule. The schedule used here is an approximation--the learning rate is decreased following a linear progression.

A problem with gradient optimization methods such as Backpropagation is that the fitting procedure may not find the global minima of the error-loss function. A momentum term is included to help escape from a local minima of the error-loss function.

###### Architecture

The idea behind a deep neural network is to pass the data through several non-linear transformations. Hierarchical representations of the data start to form in the deeper layers.

Neural networks using only sigmoidal units suffer from the vanishing gradient problem where the backpropagated signal becomes smaller with each layer it passes through. After three layers the error is almost zero. An infeasible number of updates would be required to train such a neural network. Rectified linear units have been introduced to overcome this problem. Rectified linear units are linear when the input is positive but zero everywhere else. The magnitude of the backpropagated signal does not vanish because of the neuron's linear compoonent, but the nonlinearity still allows for the units to shape the boundaries between different classes in the data. A smooth generalization of the rectified linear unit is used, called a Softmax unit.

The output of a neuron is binary response. A softmax unit is used to choose between more than two answers. Softmax units are a multinomial generalization of a neuron capable of expressing more than two outcomes.

The architecture of the neural network is detailed in the Table below.

| Layer | Neuron Type | Purpose                  | Number |
| :----:|:-----------:|:-------------------------|:------:|
| 1     | Sigmoid     | Normalize Features       | 28^2   |
| 2     | Softplus    | Nonlinear Transformation | 500    |
| 3     | Softplus    | Nonlinear Transformation | 500    |
| 4     | Softplus    | Nonlinear Transformation | 500    |
| 5     | Softmax     | Decision Layer           | 1      |

###### Regularization

The neural network contains nearly a million parameters making in prone to overfitting. Dropout is a powerful method for regularization. At each iteration, neurons are removed from the neural network with a probability of 50%. The thinned out neural network is then trained using Backpropagation. During the next iteration, all the neurons are restored and the dropout procedure is repeated to thin the neural network again removing a new set of neurons. The neural network effectively learns how to classify with approximately half of the neurons missing. Once training is complete, the weights are scaled back by 50% so that all the neurons can be used at the same time. The dropout procedure is equivalent to averaging together an exponential number of models together as one using the geometric mean.

* Why no validation set?

###### References
