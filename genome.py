# This is a genome object :
# The genome object can be best thought of
# as a Person whose brain is a neural network
# its body shall contain information like mutating a gene
# setting of new weights

import random
from deep_net_test import DeepNet 
import numpy as np 


class Genome(object):
	"""docstring for Genome"""
	def __init__(self):
		self.brain = DeepNet(2,1,0.1,1,[3])
		self.fitness = 0

	def get_weights(self):
		return self.brain.get_weights()

	def set_weights(self,new_weights):
		self.brain.set_weights(new_weights)

	def mutate(self):
		init_weights = self.brain.get_weights()
		new_weights = list(map(lambda y:y-10,list(map(lambda x:x*20,init_weights))))
		self.brain.set_weights(new_weights)

	def predict(self,value) :
		return self.brain.test(value)	