
import random
import matplotlib.pyplot as plt
import string

import cPickle
import gzip
import os
import sys
import time

import numpy as np
from logistic_sgd import load_data


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
        self.error = 0;

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
                error += output_deltas[:,k]*self.wo[j][k]
            hidden_deltas[j] = self.ah[:,j]*(1-self.ah[:,j])*error

        # update output weights and hidden weights
        for j in range(self.nh):
        #    self.wo[j,:] = self.wo[j,:]+learning_rate_alpha*output_deltas*self.ah[:,j]
            for k in range(self.no):
                self.wo[j][k] = self.wo[j][k] + learning_rate_alpha*output_deltas[:,k]*self.ah[:,j]


        for i in range(self.ni):
        #    self.wi[i,:] = self.wi[i,:]+learning_rate_beta*hidden_deltas*self.ai[:,i]
            for j in range(self.nh):
                self.wi[i][j] = self.wi[i][j] + learning_rate_alpha*hidden_deltas[j,:]*self.ai[:,i]

        #update output biases and hidden biases
        self.nhb = self.nhb + learning_rate_beta*hidden_deltas.T
        self.nob = self.nob + learning_rate_beta*output_deltas.T

        # mean square error
        error = np.sum(0.5*(targets-self.ao)**2)
        return error

    def test(self, patterns):
        """
        temporal test code for demo development
        :param patterns: test pattern
        :return:
        """
        for p in patterns:
            print(p[0], '->', self.Forward(p[0]))


    def train(self, patterns_input,pattern_out, iterations=4000, learning_rate_alpha=0.4, learning_rate_beta=0.1):
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
            for p_in,l_in in zip(patterns_input,pattern_out):
                label = mnist_binary_label(l_in)
                self.Forward(p_in)
                error +=  self.Backward(label, learning_rate_alpha, learning_rate_beta)
            if i % 100 == 0:
                print('error %-.5f' % error)


def local_load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    return test_set,valid_set,test_set

def mnist_binary_label(label_integer):
    if label_integer==0:
        return [1,0,0,0,0,0,0,0,0,0]
    elif label_integer==1:
        return [0,1,0,0,0,0,0,0,0,0]
    elif label_integer==2:
        return [0,0,1,0,0,0,0,0,0,0]
    elif label_integer==3:
        return [0,0,0,1,0,0,0,0,0,0]
    elif label_integer==4:
        return [0,0,0,0,1,0,0,0,0,0]
    elif label_integer==5:
        return [0,0,0,0,0,1,0,0,0,0]
    elif label_integer==6:
        return [0,0,0,0,0,0,1,0,0,0]
    elif label_integer==7:
        return [0,0,0,0,0,0,0,1,0,0]
    elif label_integer==8:
        return [0,0,0,0,0,0,0,0,1,0]
    elif label_integer==9:
        return [0,0,0,0,0,0,0,0,0,1]
    else:
        return [0,0,0,0,0,0,0,0,0,0]




def test_bp(n_epochs=1000,dataset='mnist.pkl.gz',n_hidden=500,batch_size=600,batch_num = 10 ):
    # Teach network XOR function
    train_dataset, valid_dataset,test_dataset = local_load_data(dataset)
    rng = np.random.RandomState(1234)
    n_input = 28*28;
    n_output = 10
    # create a network with two input, two hidden, and one output nodes
    ann_BP = BackPropagation(rng, n_input, n_hidden, n_output)

    done_looping = False
    epoch = 0
    test_score = 0.
    start_time = time.clock()

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        ann_BP.train(train_dataset[0][0:batch_size],train_dataset[1][0:batch_size])



    end_time = time.clock()

if __name__ == '__main__':
    test_bp()