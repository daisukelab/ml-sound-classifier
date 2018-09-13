import sys
sys.path.append('../..')
from common import *

import argparse

parser = argparse.ArgumentParser(description='Keras to Tensorflow converter')
parser.add_argument('model_type', type=str,
                    help='Model type "alexnet" or "mobilenetv2".')
parser.add_argument('keras_weight', type=str,
                    help='Keras model weight file.')
parser.add_argument('out_prefix', type=str,
                    help='Prefix of your tensorflow model name.')
args = parser.parse_args()
print(args, dir(args))

# create model
conf.model = args.model_type
model = create_model(conf)
model.load_weights(args.keras_weight)

# load tensorflow and keras backend
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras import backend as K
ksess = K.get_session()
print(ksess)

# transform keras model to tensorflow graph
# the output will be json-like format
K.set_learning_phase(0)
graph = ksess.graph
kgraph = graph.as_graph_def()
print(kgraph)

import os
num_output = 1
prefix = "output"
pred = [None]*num_output
outputName = [None]*num_output
for i in range(num_output):
    outputName[i] = prefix + str(i)
    pred[i] = tf.identity(model.get_output_at(i), name=outputName[i])
print('output name: ', outputName)

# convert variables in the model graph to constants
constant_graph = graph_util.convert_variables_to_constants(
    ksess, ksess.graph.as_graph_def(), outputName)

# save the model in .pb and .txt
output_dir = "./"
output_graph_name = args.out_prefix+".pb"
output_text_name = args.out_prefix+".txt"
graph_io.write_graph(constant_graph, output_dir, output_graph_name, as_text=False)
graph_io.write_graph(constant_graph, output_dir, output_text_name, as_text=True)
print('saved graph .pb at: {0}\nsaved graph .txt at: {1}'.format(
        os.path.join(output_dir, output_graph_name),
        os.path.join(output_dir, output_text_name)))
