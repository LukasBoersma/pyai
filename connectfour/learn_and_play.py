import connectfour as cf
import samples

import neural
import random
import play

import argparse

parser = argparse.ArgumentParser("Creates a NN with given properties")
parser.add_argument("--layer_sizes", help="list of layer sizes [n0, n1,...]")
parser.add_argument("--learning_rate",type=float, default=0.01)
parser.add_argument("--momentum",type=float, default=0.5)
parser.add_argument("--epochs", help="specifies number of epochs for learning",type=int,default=1)
parser.add_argument("--training_file", help="training data file name", default="data500.gz")
parser.add_argument("--ai_file", help="load AI from given file")

args = parser.parse_args()

#set default values to known parameters
total_field_size = cf.SIZE_X*cf.SIZE_Y
layer_sizes = [total_field_size*3, total_field_size*2, total_field_size,cf.SIZE_X]
if args.layer_sizes:
    print("customn layer size")
    layer_sizes = args.layer_sizes

#create AI
ai = neural.NeuralAI(layer_sizes=layer_sizes, learning_rate=args.learning_rate, momentum=args.momentum)
#ai = ai.read_model_data("best_ai")

if args.ai_file:
    print("reading")
    ai.read_model_data(args.ai_file)
else:
    td = samples.load(args.training_file)
    print("learning")
    ai.learn_epoch(td,args.epochs)


def shuffle():
    print("shuffling")
    random.shuffle(td)

def learn():
    print("learning")
    ai.learn(td)

def save():
    ai.write_model_data("ai")

def read():
    print("reading")
    ai.read_model_data("ai")

play.play(ai, 100, clever=True)

#learn_new = True
#
# if learn_new:
#
#     best_score = -10
#     clever = False
#
#     for epoch in range(30000):
#         print("============================")
#         print(">>>>>    Epoch %d" % epoch)
#         print("============================")
#         print("(current best " + str(best_score) + ")")
#
#         #shuffle()
#         #learn()
#         ai.evolve()
#         if best_score > 75 and not clever:
#             clever = True
#             best_score = -1
#
#         current_score = play_parallel(ai, 300, clever)
#         # if current_score > best_score:
#         #     print("New record")
#         #     ai.write_model_data("best_ai_t.bin")
#         #     best_score = current_score
#         # else:
#         #     ai.read_model_data("best_ai_t.bin")
#
# # else:
# #     read()
# #     play()
