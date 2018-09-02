# Predict from a sound file
#
# This is simplest premitive example to predict from a sound file.
# Audio content will be just split and predict for each split portions.
# No ensemble applied, sometimes weak to make predictions.
#
# Example:
# $ CUDA_VISIBLE_DEVICES= python premitive_file_predictor.py sample/fireworks.wav
#

from common import *
import argparse

parser = argparse.ArgumentParser(description='Run sound classifier')
parser.add_argument('audio_file', type=str,
                    help='audio file to predict.')
parser.add_argument('--model-pb-graph', '-pb', default='model/mobilenetv2_fsd2018_41cls.pb', type=str,
                    help='Feed model you want to run')
args = parser.parse_args()

model = KerasTFGraph(args.model_pb_graph,
    input_name='import/input_1',
    keras_learning_phase_name='import/bn_Conv1/keras_learning_phase',
    output_name='import/output0')

X = load_sample_as_X(conf, args.audio_file)

preds = model.predict(X)
for pred in preds:
    result = np.argmax(pred)
    print(conf.labels[result], pred[result])
