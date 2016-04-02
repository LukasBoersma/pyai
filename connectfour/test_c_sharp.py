import json
import numpy
import gzip

import neural

def load_samples(filename):
    #if '.gz' in filename:
    if filename.endswith('.gz'):
        f = gzip.open(filename,'r')
    else:
        f = open(filename,'r')
    return json.load(f)

def test_network_evaluation(network_file,in_out_file):
    ai = neural.NeuralAI()

    #ai = neural.NeuralAI.load_json('Networks/network_4x3_random.json')
    ai = neural.NeuralAI.load_json(network_file)

    #samples = load_samples('Networks/samples_4x3_random.json')
    samples = load_samples(in_out_file)

    networks_differ = 0
    for i in range(len(samples['Inputs'])):
        #print(samples['Inputs'][i])
        lasagne_output = ai.evaluate_prediction( [ samples['Inputs'][i] ] )

        #print(str(lasagne_output) + ',' + str(samples['Outputs'][i]))

        if not numpy.allclose(lasagne_output, samples['Outputs'][i]):
            networks_differ+=1
    print ('Networks produces different outputs: ' + str(networks_differ))



test_network_evaluation('Networks/network_320x120x17x107_random.json.gz','Networks/samples_320x120x17x107_random.json.gz')
