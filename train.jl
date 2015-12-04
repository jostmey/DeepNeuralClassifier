##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2015-11-17
# Environment: Julia v0.4
# Purpose: Train deep feed-forward neural network as a classifier.
##########################################################################################

##########################################################################################
# Dataset
##########################################################################################

	# Load package of the MNIST dataset.
	#
	using MNIST

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
# Settings
##########################################################################################

	# Schedule for updating the neural network.
	#
	N_minibatch = 100
	N_updates = round(Int, N_datapoints/N_minibatch)*100

	# Number of neurons in each layer.
	#
	N1 = 28^2
	N2 = 500
	N3 = 500
	N4 = 500
	N5 = 10

	# Initialize neural network parameters.
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

	# Initial learning rate.
	#
	alpha = 0.001

	# Dropout probability for removing neurons.
	#
	dropout = 0.5

	# Momentum factor.
	#
	momentum = 0.75

##########################################################################################
# Macros
##########################################################################################

	# Generate mask for neuron dropout.
	#
	remove(n) = 1.0*(dropout .<= rand(n))

	# Activation functions.
	#
	sigmoid(x) = 1.0./(1.0+exp(-x))
	softplus(x) = log(1.0+exp(x))
	softmax(x) = exp(x)./sum(exp(x))

	# Derivative of each activation function given the output.
	#
	dsigmoid(y) = y.*(1.0-y)
	dsoftplus(y) = 1.0-exp(-y)
	dsoftmax(y) = y.*(1.0-y)

##########################################################################################
# Train
##########################################################################################

	# Holds change in parameters from a minibatch.
	#
	db1 = zeros(N1)
	dW12 = zeros(N1, N2)
	db2 = zeros(N2)
	dW23 = zeros(N2, N3)
	db3 = zeros(N3)
	dW34 = zeros(N3, N4)
	db4 = zeros(N4)
	dW45 = zeros(N4, N5)
	db5 = zeros(N5)

	# Track percentage of guesses that are correct.
	#
	N_correct = 0.0
	N_tries = 0.0

	# Repeatedly update parameters.
	#
	for i = 1:N_updates

		# Generate masks for thinning out neural network (dropout procedure).
		#
		r2 = remove(N2)
		r3 = remove(N3)
		r4 = remove(N4)

		# Collect multiple updates for minibatch.
		#
		for j = 1:N_minibatch

			# Randomly load item from the dataset (part of stochastic gradient descent).
			#
			k = rand(1:N_datapoints)

			x = 6.0*features[k,:]'-3.0

			z = zeros(10)
			z[round(Int, labels[k])+1] = 1.0

			# Feedforward pass for computing the output.
			#
			y1 = sigmoid(x+b1)
			y2 = softplus(W12'*y1+b2).*r2
			y3 = softplus(W23'*y2+b3).*r3
			y4 = softplus(W34'*y3+b4).*r4
			y5 = softmax(W45'*y4+b5)

			# Backpropagation for computing the gradients of the Likelihood function.
			#
			e5 = z-y5
			e4 = (W45*e5).*dsoftplus(y4).*r4
			e3 = (W34*e4).*dsoftplus(y3).*r3
			e2 = (W23*e3).*dsoftplus(y2).*r2
			e1 = (W12*e2).*dsoftmax(y1)

			# Add the errors to the minibatch.
			#
			db1 += e1
			dW12 += y1*e2'
			db2 += e2
			dW23 += y2*e3'
			db3 += e3
			dW34 += y3*e4'
			db4 += e4
			dW45 += y4*e5'
			db5 += e5

			# Update percentage of guesses that are correct.
			#
			guess = findmax(y5)[2]-1
			answer = findmax(z)[2]-1
			if guess == answer
				N_correct += 1.0
			end
			N_tries += 1.0

		end

		# Update parameters using stochastic gradient descent.
		#
		b1 += alpha*db1
		W12 += alpha*dW12
		b2 += alpha*db2
		W23 += alpha*dW23
		b3 += alpha*db3
		W34 += alpha*dW34
		b4 += alpha*db4
		W45 += alpha*dW45
		b5 += alpha*db5

		# Reset the parameter changes from the minibatch (scale by momentum factor).
		#
		db1 *= momentum
		dW12 *= momentum
		db2 *= momentum
		dW23 *= momentum
		db3 *= momentum
		dW34 *= momentum
		db4 *= momentum
		dW45 *= momentum
		db5 *= momentum

		# Decrease the learning rate (part of stochastic gradient descent).
		#
		alpha *= (N_updates-i)/(N_updates-i+1)

		# Periodic checks.
		#
		if i%100 == 0

			# Print progress report.
			#
			println("REPORT")
			println("  Batch = $(round(Int, i))")
			println("  alpha = $(round(alpha, 8))")
			println("  Correct = $(round(100.0*N_correct/N_tries, 8))%")
			println("")
			flush(STDOUT)

			# Reset percentage of guesses that are correct.
			#
			N_tries = 0.0
			N_correct = 0.0

		end

	end

	# Scale effected weights by probability of undergoing dropout.
	#
	W23 *= 1.0-dropout
	W34 *= 1.0-dropout
	W45 *= 1.0-dropout

##########################################################################################
# Save
##########################################################################################

	# Create folder to hold parameters.
	#
	mkpath("bin")

	# Save the parameters.
	#
	writecsv("bin/train_b1.csv", b1)
	writecsv("bin/train_W12.csv", W12)
	writecsv("bin/train_b2.csv", b2)
	writecsv("bin/train_W23.csv", W23)
	writecsv("bin/train_b3.csv", b3)
	writecsv("bin/train_W34.csv", W34)
	writecsv("bin/train_b4.csv", b4)
	writecsv("bin/train_W45.csv", W45)
	writecsv("bin/train_b5.csv", b5)

