# This is the starting file of the Deep net Architecture
# Below are some conventions
# n : number of input neurons
# k : number of output neurons
# h : number of hidden layers
# bi : number of neurons in i th hidden layer denoted by b[i-1]
# Wij : weights from ith to jth layer denoted by W[i-1]
# dimension of Wij = (bj,bi+1)
# total number of layers = h + 2 (h + 1(initial) + 1(final))
# total number of weights = h + 1
# this code tries stochastic gradient descent
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

class DeepNet(object):
	"""docstring for DeepNet"""
	def __init__(self,n,k,alpha = 1,h = 0,b = []):
		self.n = n
		self.k = k
		self.h = h
		self.alpha = alpha
		self.b = b

		self.b.insert(0,n)
		self.b.append(k)

		self.W = [None]*(h+1)
		self.error_x = list()
		for i in range(self.h+1) :
			self.W[i] = 20*np.random.rand(b[i+1],b[i]+1) - 10 # final layer is the output layer hence b[h] = b(h+1) = k

	def sigmoid(self,z,deriv = False):
		if(deriv == True):
			return self.sigmoid(z)*(1-self.sigmoid(z))	# if deriv == True , return the derivative of (g(x))
		else :
			return 1/(1+np.exp(-z))


	def train(self,X,Y,iterations):
		X_train = np.insert(X,0,1,axis=1)
		for i in range(iterations):
				choice = random.randint(0,X_train.shape[0]-1)
				x = X_train[choice].reshape(self.n+1,1)		# pick a random sample from the trainig set 

				# ------------- Deep net Forward Propogation Algorithm -------------
				
				a = [None]*(self.h+2)
				g = [None]*(self.h+2)
				z = [None]*(self.h+2)	# z[j] input recieved by layer j
				a_t = [None]*(self.h+2)
				z[0] = x
				a_t[0] = x
				for j in range(self.h+1) :
					#print(" Printing W[j] ")
					#print(self.W[j])
					#print(" Print z[j] ")
					#print(z[j])
					a[j+1] = np.matmul(self.W[j],z[j])
					a_t[j+1] = np.insert(a[j+1],0,1)
					g[j+1] = self.sigmoid(a[j+1])
					z[j+1] = np.insert(g[j+1],0,1)

				final_result = np.delete(z[j+1],0)

				# --------------------- Deep net Backpropogation Algorithm -------------------- 
				delta = [None]*(self.h+2)
				
				error = Y[choice] - final_result
				
				self.error_x.append([i,math.pow(error,2)])
				delta[self.h+1] = (-error)*(self.sigmoid(a_t[self.h+1],True).reshape(a_t[self.h+1].shape[0],1))
				delta[self.h+1] = np.delete(delta[self.h+1],0).reshape(self.b[self.h+1],1)
				
				for j in range(self.h,-1,-1) :
					delta[j] = np.matmul(self.W[j].transpose(),delta[j+1].reshape(delta[j+1].shape[0],1))*(self.sigmoid(a_t[j],True).reshape(a_t[j].shape[0],1))
					delta[j] = np.delete(delta[j],0).reshape(self.b[j],1)

				# -------------------------- Deep net Updation of weights -----------------------

				for i in range(self.h+1) :
					self.W[i] = self.W[i] - self.alpha*np.matmul(delta[i+1].reshape(delta[i+1].shape[0],1),z[i].transpose().reshape(1,z[i].shape[0]))

	def predict(self,z,threshold):		# pass two parameters z and threshold
		if(z >= threshold) :			# if z > threshold , predict 1
			return 1
		else :
			return 0					# else predict 0

	def test(self,X):					# Test the perceptron against a random input
		X_test = np.insert(X,0,1)		# insert the +1 for the bias term

		a = [None]*(self.h+2)
		g = [None]*(self.h+2)
		z = [None]*(self.h+2)
		z[0] = X_test
		for j in range(self.h+1) :
			a[j+1] = np.matmul(self.W[j],z[j])
			g[j+1] = self.sigmoid(a[j+1])
			z[j+1] = np.insert(g[j+1],0,1)

		final_result = np.delete(z[j+1],0)
		return final_result
		#return self.predict(final_result,0.5)

	def test_error(self,X,Y) :
		X_test = np.insert(X,0,1,axis=1)
		temp_error = 0
		for ite in range(X_test.shape[0]) :
			a = [None]*(self.h+2)
			g = [None]*(self.h+2)
			z = [None]*(self.h+2)
			z[0] = X_test[ite]
			for j in range(self.h+1) :
				a[j+1] = np.matmul(self.W[j],z[j])
				g[j+1] = self.sigmoid(a[j+1])
				z[j+1] = np.insert(g[j+1],0,1)

			final_result = np.delete(z[j+1],0)
			temp_error += math.pow((Y[ite] - final_result),2)
		final_error = temp_error/X_test.shape[0]
		self.error = final_error
		return final_error


	def plot_error(self) :
		x = [row[0] for row in self.error_x]
		y = [row[1] for row in self.error_x]
		plt.plot(x,y)
		plt.show()
		return None

	def get_weights(self) :
		return self.W

	def set_weights(self,W) :
		self.W = W
