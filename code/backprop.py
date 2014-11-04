
import random
import matplotlib.pyplot as plt
import string

import cPickle
import gzip
import os
import sys
import time

import numpy as np



# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def activation(x):
    temp = 1/(1+np.exp(-1*(x)))
    return temp

class BackPropagation:
    """Back Propagation Class
    The Back propagation is fully connected by a weight matrix and bias vector
     Classification is done by projecting data points onto a set of hyperplanes,
    """
    def __init__(self,rng, ni, nh, no):
        """
        :param rng: random value for initializaion of each variable
        :param ni: number of input unit
        :param nh: number of hidden unit
        :param no: number of output unit
        :return: NON
        """
       # number of input, hidden, and output nodes
        self.ni = ni# +1 for bias node
        self.nh = nh
        self.no = no


        # activations for nodes
        self.ah =  np.zeros((self.nh ,1),dtype=float)
        self.ao =  np.zeros((self.no,1),dtype=float)
        self.ai =  np.zeros((self.ni,1),dtype=float)

        # create weights
        self.wi = np.asarray(rng.uniform(low=-np.sqrt(1. / (self.ni + self.nh)),high=np.sqrt(1. / (self.ni + self.nh)),size=(self.ni, self.nh)), dtype=float)
        self.wo = np.asarray(rng.uniform(low=-np.sqrt(1. / (self.nh + self.no)),high=np.sqrt(1. / (self.nh + self.no)),size=(self.nh, self.no)), dtype=float)

        # create biases
        self.nhb = np.asarray(rng.uniform(low=-np.sqrt(1. / (self.ni + self.nh)),high=np.sqrt(1. / (self.ni + self.nh)),size=(1,self.nh)), dtype=float)
        self.nob = np.asarray(rng.uniform(low=-np.sqrt(1. / (self.ni + self.nh)),high=np.sqrt(1. / (self.ni + self.nh)),size=(1,self.no)), dtype=float)


    def Forward(self, inputs):
        """
        Return the activation value of the back propagation's output layer
        :param inputs:input vector
        :return: activation value of output layer
        """
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')
        self.ai = np.asarray(inputs,dtype=float)
        self.ai = self.ai.reshape(1,self.ni)
        net_hidden = np.dot(self.ai,self.wi)+self.nhb
        self.ah= activation(net_hidden)

        # output activations
        net_output = np.dot(self.ah,self.wo)+self.nob
        self.ao=activation(net_output)
        return self.ao

    def Backward(self, targets, learning_rate_alpha, learning_rate_beta):
        """
        :param targets: label value (real value)
        :param learning_rate_alpha: learning rate for weight
        :param learning_rate_beta: learning rate for bias
        :return: error from classification
        """
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = np.zeros((self.no,1),dtype=float)
        hidden_deltas = np.zeros((self.nh,1),dtype=float)
        #for k in range(self.no):

        # update output layer offset
        output_deltas = self.ao*(1-self.ao)*(targets-self.ao)

        # update hidden layer offset
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = self.ah[:,j]*(1-self.ah[:,j])*error

        # update output weights and hidden weights
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = self.wo[j][k] + learning_rate_alpha*output_deltas[k]*self.ah[:,j]
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = self.wi[i][j] + learning_rate_alpha*hidden_deltas[j]*self.ai[:,i]

        #update output biases and hidden biases
        self.nhb = self.nhb + learning_rate_beta*hidden_deltas.T
        self.nob = self.nob + learning_rate_beta*output_deltas.T

        # mean square error
        error = np.sum(0.5*(targets-self.ao)**2)
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))


    def train(self, patterns, iterations=4000, learning_rate_alpha=0.4, learning_rate_beta=0.1):
        """
        :param patterns: train data set
        :param iterations: number of iterations using whole train data
        :param learning_rate_alpha: learning rate for weight
        :param learning_rate_beta: learning rate for bias
        :return: NAN
        """
        # learning rate alpha - learning rate for weight
        # learning rate beta - learning rate for bias
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.Forward(inputs)
                error +=  self.Backward(targets, learning_rate_alpha, learning_rate_beta)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # Teach network XOR function

    rng = np.random.RandomState(1234)
    pat = [[[0.0,0.0],[0.0]],
        [[0.0,1.0],[1.0]],
       [[1.0,0.0],[1.0]],
        [[1.0,1.0],[0.0]]]

    # create a network with two input, two hidden, and one output nodes
    n = BackPropagation(rng, 2, 5, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)

if __name__ == '__main__':
    demo()