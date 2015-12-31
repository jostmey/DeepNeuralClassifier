## Description

Example scripts for a deep, feed-forward neural network have been written from scratch. No machine learning packages are used, providing an example of how to implement the underlying algorithms of an artificial neural network. The code is written in the Julia, a programming language with a syntax similar to Matlab.

The neural network is trained on the MNIST dataset of handwritten digits. On the test dataset, the neural network correctly classifies 98.42 % of the handwritten digits. The results are pretty good for a neural network that does not contain a priori knowledge about the geometric invariances of the dataset like a Convolutional Neural Network would.

## Download

* Download: [zip](https://github.com/jostmey/DeepNeuralClassifieer/zipball/master)
* Git: `git clone https://github.com/jostmey/DeepNeuralClassifier`

## Requirements

The code requires the Julia runtime environment. Instructions on how to download and install Julia are [here](http://julialang.org/). The scripts have been developed using version 0.4 and do not work on previous versions of Julia.

The MNIST dataset must be installed in the Julia environment. To add the dataset, launch `julia` and run `Pkg.add("MNIST")` at the prompt.

## Run

Training the neural network can take several days or even weeks. Set the working directory to this folder and run the following in the command line terminal.

`julia train.jl > train.out`

The neural network will save its parameters to a folder called `bin/` once training is complete. To classify the handwritten digits in the test set, run the following command.

`julia test.jl > test.out`

The percentage of correct answers will be written at the end of the text file `test.out`.

## Performance

This package is not written for speed. It is meant to serve as a working example of an artificial neural network. As such, there is no GPU acceleration. Training using only the CPU can take days or even weeks. The training time can be shortened by reducing the number of updates, but this could lead to poorer performance on the test data. Consider using an exising machine learning package when searching for a deployable solution.

## Theory

###### Training

Feed-forward neural networks are commonly trained using *backpropagation* to minimize the error between the desired and actual output. The backpropagation algorithm is an efficient method for computing the gradient of an error-loss function. The error from the output is passed backward through the weights of the neural network, multiplying the errors by the derivative of that layer. Each backward pass through a previous layer amounts to using the Chain rule (from Calculus) to compute the derivative. The error-loss is minimized by moving the weights of the neural network down the gradient. Changes are made to the weights in small, discrete steps determined by the value of the *learning rate*.

The error is computed as the cross-entropy between the desired and actual output, which is equivalent to optimizing the Likelihood function. Ideally, backpropagation would be performed on each example in the training data before updating the weights, a process that is too expensive in practice. Instead, *stochastic gradient descent* is used. With stochastic gradient descent, a subset of examples randomly drawn from the training set are used to update the weights. The procedure will preserve the true objective function provided that the learning rate is decreased at each iteration following a specific schedule [1]. The schedule used here is an approximation--the learning rate is decreased following a linear progression.

A problem with gradient optimization methods such as backpropagation is that the fitting procedure may not find the global minima of the error-loss function. A momentum term is included to help escape from local minimums.

###### Architecture

The idea behind a deep neural network is to pass the data through several non-linear transformations. Hierarchical representations of the data may form in the deeper layers. Unfortunately, deep models are challenging to train and require more computing power.

Neural networks using only sigmoidal units suffer from the vanishing gradient problem where the backpropagated signal becomes smaller with each layer. After three layers the error is almost zero. Rectified linear units have been introduced to overcome this problem [2]. Rectified linear units are linear when the input is positive but zero everywhere else. The magnitude of the backpropagated signal does not vanish because of the neuron's linear component, but the nonlinearity still makes it possible for the units to shape arbitrary boundaries between the different labelled classes. A smooth generalization of the rectified linear unit is used, called a Softplus unit.

Because the output contains more than just two answers, a simple binary neuron cannot be used to represent the output of the neural network. A softmax unit, which is a multinomial generalization of a neuron, is used instead [3].

The architecture of the neural network is detailed in the Table below.

| Layer | Neuron Type | Purpose                  | Number    |
| :----:|:-----------:|:-------------------------|:---------:|
| 1     | Sigmoid     | Normalize Features       | 28^2      |
| 2     | Softplus    | Nonlinear Transformation | 500       |
| 3     | Softplus    | Nonlinear Transformation | 500       |
| 4     | Softplus    | Nonlinear Transformation | 500       |
| 5     | Softmax     | Decision Layer           | 1 ( x 10) |

###### Regularization

The neural network contains nearly a million parameters making it prone to overfitt on small datasets. Dropout is a powerful method for regularization [4]. At each iteration, neurons are removed from the neural network with a probability of 50%. The thinned out neural network is then trained using backpropagation. During the next iteration, all the neurons are restored and the dropout procedure is repeated to thin out a different set of neurons. The neural network effectively learns how to correctly classify the data with approximately half of the neurons missing. Once training is complete, the weights are scaled back by 50% so that all the neurons can be used at the same time. The dropout procedure has been likened to averaging together an exponential number of models together using the geometric mean.

Normally the training data should be split into a training and validation set. Multiple versions of the model are then trained on training set each using different values for the learning rate, momentum factor, dropout probability, and number of updates. The model that scores the highest on the validation set is then used on the test data. The use of a validation set means that the test data is never seen while selecting the best model, which would be cheating. That said, no validation set is used in this example because the model was never refined--only one version of the model was trained. This model was then tested directly on the test data.

###### References

[comment]: # (BIBLIOGRAPHY STYLE: MLA)

1. Robbins, Herbert, and Sutton Monro. "A stochastic approximation method." The annals of mathematical statistics (1951): 400-407.
2. Glorot, Xavier, Antoine Bordes, and Yoshua Bengio. "Deep sparse rectifier neural networks." International Conference on Artificial Intelligence and Statistics. 2011.
3. Bishop, Christopher M. *Pattern recognition and machine learning.* springer, 2006.
4. Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580 (2012).
