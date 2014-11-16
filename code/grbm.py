"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Gaussian Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections using Gaussian distribution.
"""
import cPickle
import gzip
import time
from pip.commands.search import highest_version

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import rbm
import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data

class GRBM(object):
    def __init__(self,input=None, n_visible=784, n_hidden=500,W=None,hbias=None,vbias=None,np_rng = None,theano_rng = None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if np_rng is None:
            np_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.radint(2**30))

        if W is None:
            init_W = numpy.asarray(np_rng.uniform(
                low=-4*numpy.sqrt(6./(n_hidden+n_visible)),
                high=4*numpy.sqrt(6./(n_hidden+n_visible)),
                size=(n_visible,n_hidden)),
                dtype=theano.config.floatX)
            W = theano.shared(value=init_W,name='W',borrow=True)

        if hbias is None:
            hbias = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='hbias',borrow=True)
        if vbias is None:
            hbias = theano.shared(value=numpy.zeros(n_visible,dtype=theano.config.floatX),name='hbias',borrow=True)

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng


        self.params = [self.W,self.hbias, self.vbias]

    def type(self):
        return 'Gaussian Bernoulli RBM'

    def free_energy_grbm(self,v_sample):
        '''
        Function to compute the free energy
        :param v_sample: input vector
        :return: free energy value
        '''
        wx_b = T.dot(v_sample,self.W)+self.hbias
        vbias_term = 0.5*T.dot((v_sample-self.vbias),(v_sample-self.vbias).T)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)),axis=1)
        return -hidden_term-T.diagonal(vbias_term)

    def propup(self,vis):
        pre_sigmoid_activation = T.dot(vis,self.W)+self.hbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self,v0_sample):
        pre_sigmoid_h1,h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,n=1,p=h1_mean,dtype=theano.config.floatX)
        return  [pre_sigmoid_h1, h1_mean,h1_sample]

    def propdown(self,hid):
	pre_sigmoid_activation = T.dot(hid,self.W.T) + self.vbias
        #pre_sigmoid_activation = T.dot(hid,self.W.T)+self.vbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]


    def sample_v_given_h_grbm(self,h0_sample):
        '''
        This function inferes state of visible units given hidden unit
        but a different between RBM and GRBM  is to get real values.
        this function had needed a sampling prodecure(i.e. 1_sample = self.theano_rng.binomial(size=v1_mean.shape,n=1, p=v1_mean,dtype=theano.config.floatX)
        but, it is dosen't needed.
        :param h0_sample:
        :return:
        '''
        pre_sigmoid_v1,v1_mean = self.propdown(h0_sample)
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]


    def gibbs_hvh(self,h0_sample):
        '''
        THis function implements one step of Gibbs Sampling.
        starting from the hidden state
        :param h0_sample:
        :return:
        '''
        pre_sigmoid_v1,v1_mean,v1_sample = self.sample_v_given_h_grbm(h0_sample)
        pre_sigmoid_h1,h1_mean,h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1,v1_mean,v1_sample,pre_sigmoid_h1,h1_mean,h1_sample]


    def gibbs_vhv(self,v0_sample):
        '''
        This function implements one step of Gibbs sampling.
        Staring from the visible state
        :param v0_sample:
        :return:
        '''
        pre_sigmoid_h1,h1_mean,h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1,v1_mean,v1_sample = self.sample_v_given_h_grbm(h1_sample)
        return [pre_sigmoid_v1,v1_mean,v1_sample,pre_sigmoid_h1,h1_mean,h1_sample]


    def get_cost_updates(self,learning_rate=0.1,persistent=None,k=1):
        """
        This function implements one step of Constrative Divergence K or Presistent Constrative Divergence
        :param learning_rate: learning rate used to train the Gaussian RBM
        :param presistent:
        :param k:
        :return:
        """
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                    dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost_grbm(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost


    def get_reconstruction_cost_grbm(self,updates,pre_sigmoid_nv):
        rms_cost = T.mean(T.sum((self.input -  pre_sigmoid_nv)** 2, axis=1))
        return rms_cost



def test_grbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='grbm_plots',
             n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano
    :param learning_rate:  learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    :param output_folder:
    :param n_hidden:
    :return:
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    grbm = GRBM(input=x, n_visible=28 * 28,
              n_hidden=n_hidden, np_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = grbm.get_cost_updates(learning_rate=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_grbm = theano.function([index], cost,
           updates=updates,
           givens={x: train_set_x[index * batch_size:
                                  (index + 1) * batch_size]},
           name='train_grbm')

    plotting_time = 0.
    start_time = time.clock()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_grbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = Image.fromarray(tile_raster_images(
                 X=rbm.W.get_value(borrow=True).T,
                 img_shape=(28, 28), tile_shape=(10, 10),
                 tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX))

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    [presig_hids, hid_mfs, hid_samples, presig_vis,
     vis_mfs, vis_samples], updates =  \
                        theano.scan(grbm.gibbs_vhv,
                                outputs_info=[None,  None, None, None,
                                              None, persistent_vis_chain],
                                n_steps=plot_every)

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                                updates=updates,
                                name='sample_fn')

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1),
                             dtype='uint8')
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                X=vis_mf,
                img_shape=(28, 28),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1))
        # construct image

    image = Image.fromarray(image_data)
    image.save('grbm_samples.png')
    os.chdir('../')

if __name__=='__main__':
    test_grbm()
