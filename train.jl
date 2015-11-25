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
	alpha = 0.1

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

			# Load the next item from the dataset.
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

			# Backpropagation for computing the gradients.
			#
			e5 = z-y5
			e4 = (W45*e5).*dsoftplus(y4).*r4
			e3 = (W34*e4).*dsoftplus(y3).*r3
			e2 = (W23*e3).*dsoftplus(y2).*r2
			e1 = (W12*e2).*dsoftmax(y1)

			# Update change in parameters for this minibatch.
			#
			scale = alpha/N_minibatch

			db1 += scale*e1
			dW12 += (scale*y1)*e2'
			db2 += scale*e2
			dW23 += (scale*y2)*e3'
			db3 += scale*e3
			dW34 += (scale*y3)*e4'
			db4 += scale*e4
			dW45 += (scale*y4)*e5'
			db5 += scale*e5

			# Update percentage of guesses that are correct.
			#
			guess = findmax(y5)[2]-1
			answer = findmax(z)[2]-1
			if guess == answer
				N_correct += 1.0
			end
			N_tries += 1.0

		end

		# Update parameters.
		#
		b1 += db1
		W12 += dW12
		b2 += db2
		W23 += dW23
		b3 += db3
		W34 += dW34
		b4 += db4
		W45 += dW45
		b5 += db5

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

		# Linearly decrease the learning rate.
		#
		alpha *= (N_updates-i)/(N_updates-i+1)

		# Periodic checks.
		#
		if i%100 == 0

			# Print progress report.
			#
			println("REPORT")
			println("  Batch = $(round(Int, i))")
			println("  alpha = $(round(alpha, 5))")
			println("PARAMETERS")
			println("  Mean(b1) = $(round(mean(b1), 5)), Max(b1) = $(round(maximum(b1), 5)), Min(b1) = $(round(minimum(b1), 5))")
			println("  Mean(W12) = $(round(mean(W12), 5)), Max(W12) = $(round(maximum(W12), 5)), Min(W12) = $(round(minimum(W12), 5))")
			println("  Mean(b2) = $(round(mean(b2), 5)), Max(b2) = $(round(maximum(b2), 5)), Min(b2) = $(round(minimum(b2), 5))")
			println("  Mean(W23) = $(round(mean(W23), 5)), Max(W23) = $(round(maximum(W23), 5)), Min(W23) = $(round(minimum(W23), 5))")
			println("  Mean(b3) = $(round(mean(b3), 5)), Max(b3) = $(round(maximum(b3), 5)), Min(b3) = $(round(minimum(b3), 5))")
			println("  Mean(W34) = $(round(mean(W34), 5)), Max(W34) = $(round(maximum(W34), 5)), Min(W34) = $(round(minimum(W34), 5))")
			println("  Mean(b4) = $(round(mean(b4), 5)), Max(b4) = $(round(maximum(b4), 5)), Min(b4) = $(round(minimum(b4), 5))")
			println("  Mean(W45) = $(round(mean(W45), 5)), Max(W45) = $(round(maximum(W45), 5)), Min(W45) = $(round(minimum(W45), 5))")
			println("  Mean(b5) = $(round(mean(b5), 5)), Max(b4) = $(round(maximum(b5), 5)), Min(b4) = $(round(minimum(b5), 5))")
			println("SCORE")
			println("  Correct = $(round(100.0*N_correct/N_tries, 5))%")
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

