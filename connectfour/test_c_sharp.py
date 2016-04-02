import json
import numpy
import gzip

import neural


def test_network_evaluation(network_file,in_out_file):
    ai = neural.NeuralAI()

    #ai = neural.NeuralAI.load_json('Networks/network_4x3_random.json')
    ai = neural.NeuralAI.load_json(network_file)

    #samples = load_samples('Networks/samples_4x3_random.json')
    samples = neural.NeuralAI.load_samples_json(in_out_file)

    networks_differ = 0
    for i in range(len(samples['Inputs'])):
        #print(samples['Inputs'][i])
        lasagne_output = ai.evaluate_prediction( [ samples['Inputs'][i] ] )

        #print(str(lasagne_output) + ',' + str(samples['Outputs'][i]))

        if not numpy.allclose(lasagne_output, samples['Outputs'][i]):
            networks_differ+=1
    print ('Networks produces different outputs: ' + str(networks_differ))

network_file_pre_learning = 'Networks/sign_network_before_training.json.gz'
network_file_post_learning='Networks/sign_network_after_one_sample.json.gz'
#training_file = 'Networks/sign_training.json.gz'
training_file = 'Networks/sign_one_sample.json.gz'
test_file = 'Networks/sign_test_correct.json.gz'

ai = neural.NeuralAI(momentum=0)
ai = neural.NeuralAI.load_json(network_file_pre_learning)
ai_learned_c = neural.NeuralAI.load_json(network_file_post_learning)

#W_py = ai.l_out.W.get_value()
#W_cs = ai_learned_c.l_out.W.get_value()
# print (W_py)
# print (ai.l_out.b.get_value())
# print (W_cs)

# import math
# def sigmoid(x):
#   return 1 / (1 + math.exp(-x))

ai.learn_json(training_file)
# training_size = 5000
# td = []
# for i in range(training_size):
#     input = numpy.random.uniform(-1,1)
#
#     td.append( [ [i],[sigmoid(i)] ] )
# ai.learn_epoch(td,40)

for i in range(len(ai.l_hidden)):
    W_py = ai.l_hidden[i].W.get_value()
    W_cs = ai_learned_c.l_hidden[i].W.get_value()
    # print (W_py)
    # print (ai.l_out.b.get_value())
    # print (W_cs)
    if numpy.allclose(W_cs, W_py):
        print('Hidden layer ' +str(i) + ' equal!')
    else:
        print("Weight matrices are not equal! Difference in layer " + str(i))

W_py = ai.l_out.W.get_value()
W_cs = ai_learned_c.l_out.W.get_value()
# print (W_py)
# print (ai.l_out.b.get_value())
# print (W_cs)
if numpy.allclose(W_cs, W_py):
    print("Output layer equal!")
else:
    print("Weight matrices are not equal! Difference in output layer ")


test_data = neural.NeuralAI.load_samples_json(test_file)
merged_in_out = zip(test_data['Inputs'],test_data['Outputs'])

fails = 0
right = 0
for inp,outp in merged_in_out:
    net_outp = ai.evaluate(inp)
    #print(net_outp)
    #print(outp)
    if abs(net_outp[0] - outp[0]) > 0.3:
        fails +=1
#        print("Fail")
    else:
        right += 1
#        print("Right")
print ("Nr of fails:  " + str(fails))
print ("Nr of correct:" + str(right))
