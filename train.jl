##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-11-17
# Environment: Julia v0.4
# Purpose: Train deep feed-forward neural network as a classifier.
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
	data = traindata()

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
# Parameters
##########################################################################################

	# Schedule for updating the neural network.
	#
	N_minibatch = 10
	N_updates = round(Int, N_datapoints/N_minibatch)

	# Number of neurons in each layer.
	#
	N1 = 28^2
	N2 = 500
	N3 = 500
	N4 = 10

	# Neural network parameters.
	#
	b1 = 0.1*randn(N1)
	W12 = 0.1*randn(N1, N2)
	b2 = 0.1*randn(N2)
	W23 = 0.1*randn(N2, N3)
	b3 = 0.1*randn(N3)
	W34 = 0.1*randn(N3, N4)
	b4 = 0.1*randn(N4)

	# Initial learning rate.
	#
	alpha = 0.01

	# Dropout probability for removing a neuron.
	#
	dropout = 0.5

	# Momentum factor.
	#
	momentum = 0.75

##########################################################################################
# Globals
##########################################################################################

	# Activation functions.
	#
	sigmoid(x) = 1.0./(1.0+exp(-x))
	softplus(x) = log(1.0+exp(x))
	softmax(x) = exp(x)./sum(exp(x))

	# Derivative of each activation function given the output.
	#
	D_sigmoid(y) = y.*(1.0-y)
	D_softplus(y) = 1.0-exp(-y)
	D_softmax(y) = y.*(1.0-y)

	# Variables for storing parameter changes.
	#
	db1 = zeros(N1)
	dW12 = zeros(N1,N2)
	db2 = zeros(N2)
	dW23 = zeros(N2,N3)
	db3 = zeros(N3)
	dW34 = zeros(N3,N4)
	db4 = zeros(N4)

##########################################################################################
# Train
##########################################################################################

	# Index for training trial.
	#
	k = 1

	# Track percentage of guesses that are correct.
	#
	N_correct = 0.0
	N_tries = 0.0

	# Parameters updated each cycle.
	#
	for i = 1:N_updates

		# Collect updates each cycle for minibatch.
		#
		for j = 1:N_minibatch

			# Load the input and target output.
			#
			x = 6.0*features[k,:]'-3.0
			z = zeros(10) ; z[labels[k]+1] = 1.0

			# Feedforward pass for computing the output.
			#
			y1 = sigmoid(x+b1)
			y2 = softplus(W12'*y1+b2)
			y3 = softplus(W23'*y2+b3)
			y4 = softmax(W34'*y3+b4)

			# Backpropagation for computing the gradients.
			#
			e4 = z-y4
			e3 = (W34*e4).*D_softplus(y3)
			e2 = (W23*e3).*D_softplus(y2)
			e1 = (W12*e2).*D_softmax(y1)

			# Collect minibatch of updates.
			#
			db1 += (alpha/N_minibatch)*e1
#			dW12 += (alpha/N_minibatch)*(y1*e2')
BLAS.gemm!('N', 'T', alpha/N_minibatch, y1, e2, 1.0, dW12)	# BLAS package faster at calculating outer product.
			db2 += (alpha/N_minibatch)*e2
#			dW23 += (alpha/N_minibatch)*(y2*e3')
BLAS.gemm!('N', 'T', alpha/N_minibatch, y2, e3, 1.0, dW23)	# BLAS package faster at calculating outer product.
			db3 += (alpha/N_minibatch)*e3
#			dW34 += (alpha/N_minibatch)*(y3*e4')
BLAS.gemm!('N', 'T', alpha/N_minibatch, y3, e4, 1.0, dW34)	# BLAS package faster at calculating outer product.
			db4 += (alpha/N_minibatch)*e4

			# Update percentage of guesses that are correct.
			#
			N_tries += 1.0
			if findmax(z)[2] == findmax(y4)[2]
				N_correct += 1.0
			end

			# Move index for training trial to its next value.
			#
			k = (k < N_datapoints) ? k+1 : 1

		end

		# Update the parameters.
		#
		b1 += db1
		W12 += dW12
		b2 += db2
		W23 += dW23
		b3 += db3
		W34 += dW34
		b4 += db4

		# Scale previous parameter changes by the momentum factor.
		#
		db1 *= momentum
		dW12 *= momentum
		db2 *= momentum
		dW23 *= momentum
		db3 *= momentum
		dW34 *= momentum
		db4 *= momentum

		# Adjust the learning rate.
		#
		alpha = alpha*(N_updates-i)/(N_updates-i+1)

		# Periodically print a progress report.
		#
		if i%100 == 0

			println("REPORT")
			println("  Batch = $(round(Int, i)")
			println("  alpha = $(round(alpha, 5))")
			println("PARAMETERS")
			println("  Mean(b1) = $(round(mean(b1),5)), Max(b1) = $(round(maximum(b1),5)), Min(b1) = $(round(minimum(b1),5))")
			println("  Mean(W12) = $(round(mean(W12),5)), Max(W12) = $(round(maximum(W12),5)), Min(W12) = $(round(minimum(W12),5))")
			println("  Mean(b2) = $(round(mean(b2),5)), Max(b2) = $(round(maximum(b2),5)), Min(b2) = $(round(minimum(b2),5))")
			println("  Mean(W23) = $(round(mean(W23),5)), Max(W23) = $(round(maximum(W23),5)), Min(W23) = $(round(minimum(W23),5))")
			println("  Mean(b3) = $(round(mean(b3),5)), Max(b3) = $(round(maximum(b3),5)), Min(b3) = $(round(minimum(b3),5))")
			println("  Mean(W34) = $(round(mean(W34),5)), Max(W34) = $(round(maximum(W34),5)), Min(W34) = $(round(minimum(W34),5))")
			println("  Mean(b4) = $(round(mean(b4),5)), Max(b4) = $(round(maximum(b4),5)), Min(b4) = $(round(minimum(b4),5))")
			println("ACCURACY")
			println("  Correct = $(round(100.0*N_correct/N_tries, 5))%")
			println("")
			flush(STDOUT)

			N_tries = 0.0
			N_correct = 0.0

		end

	end

##########################################################################################
# Save
##########################################################################################

	# Create folder to hold parameters.
	#
	mkdir("bin/")

	# Save the parameters.
	#
	writecsv("bin/train_b1.csv", b2)
	writecsv("bin/train_W12.csv", W12)
	writecsv("bin/train_b2.csv", b2)
	writecsv("bin/train_W23.csv", W23)
	writecsv("bin/train_b3.csv", b3)
	writecsv("bin/train_W34.csv", W34)
	writecsv("bin/train_b4.csv", b4)


