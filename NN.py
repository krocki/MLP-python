# -*- coding: utf-8 -*-
# @Author: krocki
# @Date:   2016-12-21 10:21:06
# @Last Modified by:   krocki
# @Last Modified time: 2016-12-22 16:55:06
#
# a simple implementation of a feedforward NN

import numpy as np

class NN:

	layers = [] # stack of layers

	def __init__(self, layers):
		self.layers = layers
		self.num_layers = len(self.layers)

	def forward(self, inputs):
		self.layers[0].x = inputs
		for i in range(self.num_layers):
			self.layers[i].forward()
			if i < self.num_layers - 1:
				self.layers[i+1].x = self.layers[i].y;
			
		return self.layers[-1].y

	def backward(self, targets):
		self.layers[-1].dy = targets
		for i in reversed(range(self.num_layers)):
			self.layers[i].resetgrads();
			self.layers[i].backward()
			if i > 0:
				self.layers[i-1].dy = self.layers[i].dx;
		return self.layers[0].dx

	def adapt(self, alpha):
		for i in range(self.num_layers):
			self.layers[i].applygrads(alpha);

class Layer:

	# inputs, outputs, input grads, output grads
	x, y, dx, dy = [], [], [], []

	def __init__(self, inputs, outputs, batchsize):
		self.x = np.zeros((inputs, batchsize), dtype=np.float)
		self.y = np.zeros((outputs, batchsize), dtype=np.float)
		self.dx = np.zeros_like(self.x, dtype=np.float)
		self.dy = np.zeros_like(self.y, dtype=np.float)

	def forward(self):
		raise NotImplementedError()

	def backward(self):
		raise NotImplementedError()

	def resetgrads(self):
		pass

	def applygrads(self, alpha):
		pass

class Linear(Layer):

	# weights x-y, biases, w grads, b grads
	W, b, dW, db = [], [], [], []

	def __init__(self, inputs, outputs, batchsize):
		Layer.__init__(self, inputs, outputs, batchsize)
		self.W = 0.1 * np.random.randn(outputs, inputs)
		self.b = np.zeros((outputs, 1), dtype=np.float)
		self.resetgrads();

	def forward(self):
		self.y = np.dot(self.W, self.x) + self.b;

	def backward(self):
		self.dW = np.dot(self.dy, self.x.T)
		self.db = np.expand_dims(np.sum(self.dy, axis=1), axis=1)
		self.dx = np.dot(self.W.T, self.dy)

	def resetgrads(self):
		self.dW = np.zeros_like(self.W, dtype=np.float)
		self.db = np.zeros_like(self.db, dtype=np.float)

	def applygrads(self, alpha):
		self.b += self.db * alpha
		self.W += self.dW * alpha

class Softmax(Layer):

	def __init__(self, inputs, outputs, batchsize):
		Layer.__init__(self, inputs, outputs, batchsize)

	def forward(self):
		self.y = softmax(self.x);

	def backward(self):
		self.dx = self.dy - self.y;

class ReLU(Layer):

	def __init__(self, inputs, outputs, batchsize):
		Layer.__init__(self, inputs, outputs, batchsize)

	def forward(self):
		self.y = rectify(self.x);

	def backward(self):
		self.dx = drectify(self.y) * self.dy;

#helper functions
def rectify(x):
	return np.maximum(x, 0);

def drectify(x):
	return x > 0;

#probs(class) = exp(x, class)/sum(exp(x, class))
def softmax(x):
	e = np.exp(x);
	sums = np.sum(e, axis=0);
	return e / sums;

#cross-entropy
def crossentropy(predictions, targets):
	return -np.sum (np.log(np.sum(targets * predictions, axis=0)))
