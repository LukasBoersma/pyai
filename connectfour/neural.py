import lasagne as L
import theano as T
import numpy


import cPickle as pickle
import os


import connectfour as cf

class NeuralAI:

    def __init__(self):
        # create network
        total_field_size = cf.SIZE_X*cf.SIZE_Y
        self.input_var = T.tensor.matrix('input_var')
        self.l_in = L.layers.InputLayer(shape=(1,3*total_field_size), input_var = self.input_var)
        self.l_hidden1 = L.layers.DenseLayer(self.l_in, num_units= total_field_size*8 )
        self.l_hidden2 = L.layers.DenseLayer(self.l_hidden1, num_units= total_field_size )
        self.l_hidden3 = L.layers.DenseLayer(self.l_hidden2, num_units= cf.SIZE_X*4 )
        self.l_out = L.layers.DenseLayer(self.l_hidden3, num_units=cf.SIZE_X)

        # define a function f that maps an input input_var to an output reaction

        # y is a function that evaluates the network with final layer l_out and input input_var
        self.prediction = L.layers.get_output(self.l_out)
        self.evaluate_prediction = T.function([self.input_var], self.prediction)

        # collect all parameters that can be trained
        self.params = L.layers.get_all_params(self.l_out, trainable=True)

        self.score = T.tensor.vector('targets')
        self.loss = L.objectives.squared_error(self.prediction, self.score)

        self.loss = self.loss.mean()

        self.updates = L.updates.nesterov_momentum(self.loss, self.params, learning_rate=0.01, momentum=0.9)
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
        
    def learn(self, training_data):
        
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