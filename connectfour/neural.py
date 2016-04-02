import lasagne as L
import theano as T
import numpy
import gzip
import random
import json

import cPickle as pickle
import os


import connectfour as cf

class NeuralAI:

    def __init__(self,
    layer_sizes= [3*cf.SIZE_X*cf.SIZE_Y,84,42,cf.SIZE_X],
    learning_rate = 0.01,
    momentum = 0.5,
    batch_size = 1):
        # create network
        total_field_size = cf.SIZE_X*cf.SIZE_Y
        self.input_var = T.tensor.matrix('input_var')
        if len(layer_sizes) < 2:
            raise Exception("You need to specify at least 2 layers")
        self.l_in = L.layers.InputLayer(shape=(batch_size,layer_sizes[0]), input_var = self.input_var)

        self.l_hidden = []
        #in two-layer case the output layer is connected directly to the input layer
        self.l_out = L.layers.DenseLayer(self.l_in, num_units=layer_sizes[-1])
        if len(layer_sizes) > 2:
            #first hidden layer is special as it is connected to input layer
            self.l_hidden.append( L.layers.DenseLayer(self.l_in, num_units= layer_sizes[1] ) )
            # three layer have been constructed so far:
            # input, first hidden layer, output (output will be changed later)
            for l in range(2,len(layer_sizes)-1):
                self.l_hidden.append(L.layers.DenseLayer(self.l_hidden[-1], num_units= layer_sizes[l] ))

            self.l_out = L.layers.DenseLayer(self.l_hidden[-1], num_units=layer_sizes[-1])

        # define a function f that maps an input input_var to an output reaction

        # y is a function that evaluates the network with final layer l_out and input input_var
        self.prediction = L.layers.get_output(self.l_out)
        self.evaluate_prediction = T.function([self.input_var], self.prediction)

        # collect all parameters that can be trained
        self.params = L.layers.get_all_params(self.l_out, trainable=True)

        self.score = T.tensor.vector('targets')
        self.loss = L.objectives.squared_error(self.prediction, self.score)

        self.loss = self.loss.mean()

        self.updates = L.updates.nesterov_momentum(self.loss, self.params, learning_rate=learning_rate, momentum=momentum)
        self.train_fn = T.function([self.input_var, self.score], self.loss, updates=self.updates)

    def evolve(self):
        matrices = L.layers.get_all_param_values(self.l_out)

        mean_val = numpy.mean( [numpy.mean(item.flatten()) for item in matrices])

        scale_factor = numpy.absolute(mean_val/15)
        print ("Deviation: " + str(scale_factor) )

        for m in matrices:
            for entry in m:
                entry += numpy.random.normal(scale=scale_factor)
        L.layers.set_all_param_values(self.l_out, matrices)

    def learn(self, training_data,silent = False):

        if not silent:
            print(len(training_data))

        loss_function = T.function([self.input_var, self.score], self.loss)



        for (field, score ) in training_data:
            #print("learning")
            #print(len(field))
            #print(field)
            #print(score)

            #print( "loss before fit: " + str(loss_function([field],score ))  )
            self.train_fn([field],score)
            #print( "loss after fit: " + str(loss_function([field],score ))  )

    def learn_epoch(self, training_data, epochs):
        print("learning " + str(epochs) + " epochs")
        for i in range(epochs):
            random.shuffle(training_data)
            self.learn(training_data, silent = True)

    # see https://gist.github.com/senbon/70adf5410950c0dc882b
    # and https://github.com/Lasagne/Lasagne/issues/7
    def move(self, field):
        #print("moving")
        #print(len(field))
        #print(field)
        score = self.evaluate_prediction([field])

        return numpy.argmax(score)


    def read_model_data(self, filename):
        #"""Unpickles and loads parameters into a Lasagne model."""
        #filename = os.path.join('./', '%s.%s' % (filename, PARAM_EXTENSION))
        with open(filename, 'r') as f:
            data = pickle.load(f)
        L.layers.set_all_param_values(self.l_out, data)


    def write_model_data(self, filename):
        #"""Pickels the parameters within a Lasagne model."""
        data = L.layers.get_all_param_values(self.l_out)
        #filename = os.path.join('./', filename)
        #filename = '%s.%s' % (filename, PARAM_EXTENSION)
        with open(filename, 'w') as f:
            pickle.dump(data, f)

    def pickle(self, filename):
        f = gzip.open(filename, 'wb')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def unpickle(filename):
        f = gzip.open(filename, 'rb')
        pickle.load(f)
        f.close()

    @staticmethod
    def load_json(filename):
        f = open(filename,'r')
        dic = json.load(f)

        depth = len(dic['Layers'])
        layer_sizes = []
        for i in range(depth):
            layer_sizes.append(dic['Layers'][i]['Size'])

        print('Layer sizes' + str(layer_sizes) )

        # mild warning, this learning parameters could cause problems when comparing this implementation to c#
        ai = NeuralAI(layer_sizes=layer_sizes,learning_rate=0.01,momentum=0,batch_size=1)

        #import ipdb;ipdb.set_trace()
        # sets weights in last layer of network
        #print(ai.l_out.W.get_value())
        new_weights = numpy.reshape(dic['Layers'][1]['Weights']['Values'], (layer_sizes[-2],layer_sizes[-1]),'C')
        #print (new_weights)
        ai.l_out.W.set_value(new_weights)
        #print(ai.l_out.W.get_value())

        #sets bias in last layer of network
        new_bias = [0]*layer_sizes[-1]
        ai.l_out.b.set_value(new_bias)
        print(ai.l_out.b.get_value())
        return ai
