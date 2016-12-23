# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-21 20:36:59
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-22 22:39:47
import cPickle, gzip, numpy
from NN import *

# MNIST # Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

## set up NN
batchsize = 100
learningrate = 1e-3
epochs = 1000
batches_per_epoch = 10
num_classes = 10
smoothloss = np.log(num_classes)
accuracy = 100/num_classes;

# should reach about 97-98 % accuracy on the test set and ~0.03 training loss

layers = [ 
			Linear(784, 256, batchsize), 
			ReLU(256, 256, batchsize), 
			Linear(256, 64, batchsize), 
			ReLU(64, 64, batchsize), 
			Linear(64, 10, batchsize), 
			Softmax(10, 10, batchsize)
		]

nn = NN(layers)

def makebatch(xs, batchsize, num_classes):
	rands = numpy.random.randint(0, high=xs[0].shape[0], size=batchsize)
	inputs = xs[0][rands, :].T
	targets_int = xs[1][rands]
	targets = np.eye(num_classes)[targets_int].T
	return inputs, targets, targets_int

# make batch
for e in range(epochs):
	for i in range(batches_per_epoch):
		inputs, targets, targets_int = makebatch(train_set, batchsize, num_classes)
		outputs = nn.forward(inputs)
		nn.backward(targets)
		nn.adapt(learningrate)
		smoothloss = 0.9 * smoothloss + 0.1 * crossentropy(outputs, targets)/batchsize

	#test
	inputs, targets, targets_int = makebatch(test_set, batchsize, num_classes)
	outputs = nn.forward(inputs)
	predicted_classes = np.argmax(outputs, axis=0)
	accuracy = 0.9 * accuracy + 0.1 * np.sum(targets_int == predicted_classes)*100/batchsize
	print "epoch %3d" % e + " --- training loss: " + '%.3f' % smoothloss + ", test accuracy: " + '%.1f %%' % accuracy

