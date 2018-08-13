# Common functions and definitions
#
# This file defines commonly used parts for ease of programming.
# Import as follows:
#
# > from common import *
#
# Private notation(s):
# - mels = melspectrogram
#

# # Basic definitions
# Warning suppression
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.warnings.filterwarnings('ignore')
np.random.seed(1001)

import sys
import shutil
from pathlib import Path
import pandas as pd

# # Common Configration
# Labels and integer converter
labels = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
       'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard',
       'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing',
       'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime',
       'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica', 'Gong',
       'Microwave_oven', 'Burping_or_eructation', 'Double_bass', 'Shatter',
       'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano', 'Meow',
       'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
       'Violin_or_fiddle', 'Finger_snapping']
label2int = {l:i for i, l in enumerate(labels)}
num_classes = len(labels)

conf={}
conf['sampling_rate'] = 44100
conf['duration'] = 1
conf['hop_length'] = 347 # to make time steps 128
conf['fmin'] = 20
conf['fmax'] = conf['sampling_rate'] // 2
conf['n_mels'] = 128
conf['n_fft'] = conf['n_mels'] * 20
conf['audio_split'] = 'dont_crop'
conf['learning_rate'] = 0.0001

def auto_complete_conf(conf):
    conf['samples'] = conf['sampling_rate'] * conf['duration']
    conf['dims'] = (conf['n_mels'], 1 + int(np.floor(conf['samples']/conf['hop_length'])), 1)

auto_complete_conf(conf)

# # Model
import keras
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

def model_mobilenetv2(input_shape, num_classes):
    base_model = MobileNetV2(weights=None, input_shape=input_shape, include_top=False,
                            alpha=0.35, depth_multiplier=0.5)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_model(conf, num_classes):
    model = model_mobilenetv2(input_shape=conf['dims'], num_classes=num_classes)
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=conf['learning_rate']),
              metrics=['accuracy'])
    model.summary()
    return model

# # Audio Utilities
import librosa
import librosa.display

def read_audio(conf, pathname):
    y, sr = librosa.load(pathname, sr=conf['sampling_rate'])
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf['samples']: # long enough
        if conf['audio_split'] == 'head':
            y = y[0:0+conf['samples']]
    else: # pad blank
        padding = conf['samples'] - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf['samples'] - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf['sampling_rate'],
                                                 n_mels=conf['n_mels'],
                                                 hop_length=conf['hop_length'],
                                                 n_fft=conf['n_fft'],
                                                 fmin=conf['fmin'],
                                                 fmax=conf['fmax'])
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels):
    import IPython
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    matplotlib.style.use('ggplot')

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf['sampling_rate'], hop_length=conf['hop_length'],
                            fmin=conf['fmin'], fmax=conf['fmax'])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.show()

def read_as_melspectrogram(conf, pathname, debug_display=False):
    x = read_audio(conf, pathname)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf['sampling_rate']))
        show_melspectrogram(conf, mels)
    return mels

# # Dataset Utilities
def samplewise_mean_X(X):
    for i in range(len(X)):
        _mean, _std = np.mean(X[i]), np.std(X[i])
        X[i] -= _mean
        #X[i] /= _std + np.max(X[i]) # 1.0 # Kind of Compressor effect
        X[i] /= _std + 1.0 # Kind of Compressor effect


def split_long_data(conf, X):
    # Splits long mel-spectrogram data with small overlap
    L = X.shape[1]
    one_length = conf['dims'][1]
    loop_length = int(one_length * 0.9)
    min_length = int(one_length * 0.2)
    print(' sample length', L, 'to cut every', one_length)
    for idx in range(L // loop_length):
        cur = loop_length * idx
        rest = L - cur
        if one_length <= rest:
            yield X[:, cur:cur+one_length]
        elif min_length <= rest:
            cur = L - one_length
            yield X[:, cur:cur+one_length]

def convert_X(fnames, conf, datapath):
    """Convert all files listed on fnames and generates training set.
    Long samples are split into pieces by conf['samples'].
    
    :param fnames: list of filenames
    :param conf: configurations
    :param datapath: folder to raw samples 
    :return X: mel-spectrogram arraay, shape=(# of splits, conf('n_mels'), conf('dims')[1], 1)
    :return index_map: index mapping to original file, shape=(# of splits,)
    """
    X = []
    datapath = Path(datapath)
    index_map = []
    for i, fname in enumerate(fnames):
        print('processing', fname)
        data = read_as_melspectrogram(conf, datapath / fname)
        for chunk in split_long_data(conf, data):
            X.append(np.expand_dims(chunk, axis=-1))
            index_map.append(i)
    return np.array(X), np.array(index_map)

def mels_len(mels): return mels.shape[1]

def audio_sample_to_X(conf, norm_audio):
    mels = audio_to_melspectrogram(conf, norm_audio)
    X = []
    for s in range(0, mels_len(mels) // conf['dims'][1]):
        cur = s * conf['dims'][1]
        X.append(mels[:, cur:cur + conf['dims'][1]][..., np.newaxis])
    X = np.array(X)
    #X /= np.max(np.abs(X))   ########### REMOVE AFTER NEWLY TRAINED...
    samplewise_mean_X(X)
    return X

def load_sample_as_X(conf, filename):
    norm_audio = read_audio(conf, filename)
    return audio_sample_to_X(conf, norm_audio)

# # Tensorflow Utilities
import tensorflow as tf

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

class KerasTFGraph:
    def __init__(self, model_pb_filename, input_name,
                 keras_learning_phase_name, output_name):
        #self.model_pb_filename = model_pb_filename
        self.graph = load_graph(model_pb_filename)
        #self.input_name = input_name
        #self.keras_learning_phase_name = keras_learning_phase_name
        #self.output_name = output_name
        self.layer_in = self.graph.get_operation_by_name(input_name)
        self.leayer_klp = self.graph.get_operation_by_name(keras_learning_phase_name)
        self.layer_out = self.graph.get_operation_by_name(output_name)
    def predict(self, X):
        with tf.Session(graph=self.graph) as sess:
            tf_preds = sess.run(self.layer_out.outputs[0], 
                               {self.layer_in.outputs[0]: X,
                                self.leayer_klp.outputs[0]: 0})
        return tf_preds


