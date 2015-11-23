##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-11-17
# Environment: Julia v0.4
# Purpose: Classify test data using deep feed-forward neural network.
##########################################################################################

##########################################################################################
# Packages
##########################################################################################

	# Load the MNIST dataset of handwritten digits.
	#
	using MNIST

##########################################################################################
# Dataset
##########################################################################################

	# Load the dataset.
	#
	data = testdata()

	# Scale feature values to be between 0 and 1.
	#
	features = data[1]'
	features /= 255.0

	# Copy over the labels.
	#
	labels = data[2]

	# Size of the dataset.
	#
	N_datapoints = size(features, 1)

##########################################################################################
# Settings
##########################################################################################

	# Number of neurons in each layer.
	#
	N1 = 28^2
	N2 = 500
	N3 = 500
	N4 = 500
	N5 = 10

	# Neural network parameters.
	#
	b1 = 0.1*randn(N1)
	W12 = 0.1*randn(N1, N2)
	b2 = 0.1*randn(N2)
	W23 = 0.1*randn(N2, N3)
	b3 = 0.1*randn(N3)
	W34 = 0.1*randn(N3, N4)
	b4 = 0.1*randn(N4)
	W45 = 0.1*randn(N4, N5)
	b5 = 0.1*randn(N5)

##########################################################################################
# Macros
##########################################################################################

	# Activation functions.
	#
	sigmoid(x) = 1.0./(1.0+exp(-x))
	softplus(x) = log(1.0+exp(x))
	softmax(x) = exp(x)./sum(exp(x))

##########################################################################################
# Test
##########################################################################################

	# Print header.
	#
	println("RESPONSES")

	# Track percentage of guesses that are correct.
	#
	N_correct = 0.0
	N_tries = 0.0

	# Classify each item in the dataset.
	#
	for i = 1:N_datapoints

		# Load the input.
		#
		x = 6.0*features[i,:]'-3.0

		# Feedforward pass for computing the output.
		#
		y1 = sigmoid(x+b1)
		y2 = softplus(W12'*y1+b2)
		y3 = softplus(W23'*y2+b3)
		y4 = softplus(W34'*y3+b4)
		y5 = softmax(W45'*y4.+b5)

		# Update percentage of guesses that are correct.
		#
		guess = findmax(z)[2]-1
		answer = round(Int, labels[i])
		if guess == answer
			N_correct += 1.0
		end
		N_tries += 1.0

		# Print response.
		#
		println("  i = $(i), Guess = $(guess), Answer = $(answer)")

	end

	# Skip a line.
	#
	println("")

##########################################################################################
# Results
##########################################################################################

	# Print progress report.
	#
	println("REPORT")
	println("  Correct = $(round(100.0*N_correct/N_tries, 5))%")
	println("")

