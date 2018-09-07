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
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.warnings.filterwarnings('ignore')
np.random.seed(1001)

import sys
import shutil
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path.cwd()))

# # Configration
def auto_complete_conf(conf):
    conf.folder = Path(conf.folder)
    conf.label2int = {l:i for i, l in enumerate(conf.labels)}
    conf.num_classes = len(conf.labels)
    conf.samples = conf.sampling_rate * conf.duration
    conf.rt_chunk_samples = conf.sampling_rate // conf.rt_oversamples
    conf.mels_onestep_samples = conf.rt_chunk_samples * conf.rt_process_count
    conf.mels_convert_samples = conf.samples + conf.mels_onestep_samples
    conf.dims = (conf.n_mels, 1 + int(np.floor(conf.samples/conf.hop_length)), 1)

from config import *
auto_complete_conf(conf)
print(conf)

# # Data utilities
def load_labels(conf):
    conf.labels = loaddata(conf, 'labels.npy')
    auto_complete_conf(conf)
    print('Labels are', conf.labels)

def datapath(conf, filename):
    return conf.folder / filename

def loaddata(conf, filename):
    return np.load(conf.folder / filename)

def load_dataset(X_file, y_file, idx_file):
    return loaddata(conf, X_file), \
        keras.utils.to_categorical(loaddata(conf, y_file)), \
        loaddata(conf, idx_file)

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

def create_model(conf, num_classes, weights=None):
    model = model_mobilenetv2(input_shape=conf.dims, num_classes=num_classes)
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=conf.learning_rate),
              metrics=['accuracy'])
    if weights is not None:
        print('Loading weights:', weights)
        model.load_weights(weights, by_name=True, skip_mismatch=True)
    model.summary()
    return model

def freeze_model_layers(model, trainable_after_this=''):
    trainable = False
    for layer in model.layers:
        if layer.name == trainable_after_this:
            trainable = True
        layer.trainable = trainable

# # Audio Utilities
import librosa
import librosa.display

def read_audio(conf, pathname):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if conf.audio_split == 'head':
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
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
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.show()

def read_as_melspectrogram(conf, pathname, debug_display=False):
    x = read_audio(conf, pathname)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels

# # Dataset Utilities
def deprecated_samplewise_mean_audio_X(X):
    for i in range(len(X)):
        X[i] -= np.mean(X[i])
        X[i] /= (np.std(X[i]) + 1.0)

def samplewise_normalize_audio_X(X):
    for i in range(len(X)):
        X[i] -= np.min(X[i])
        X[i] /= (np.max(np.abs(X[i])) + 1.0)

def split_long_data(conf, X):
    # Splits long mel-spectrogram data with small overlap
    L = X.shape[1]
    one_length = conf.dims[1]
    loop_length = int(one_length * 0.9)
    min_length = int(one_length * 0.2)
    print(' sample length', L, 'to split by', one_length)
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
    Long samples are split into pieces by conf.samples.
    
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
    for s in range(0, mels_len(mels) // conf.dims[1]):
        cur = s * conf.dims[1]
        X.append(mels[:, cur:cur + conf.dims[1]][..., np.newaxis])
    X = np.array(X)
    samplewise_normalize_audio_X(X)
    return X

def load_sample_as_X(conf, filename):
    norm_audio = read_audio(conf, filename)
    return audio_sample_to_X(conf, norm_audio)

def geometric_mean_preds(_preds):
    preds = _preds.copy()
    for i in range(1, preds.shape[0]):
        preds[0] = np.multiply(preds[0], preds[i])
    return np.power(preds[0], 1/preds.shape[0])

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
        self.graph = load_graph(model_pb_filename)
        self.layer_in = self.graph.get_operation_by_name(input_name)
        self.leayer_klp = self.graph.get_operation_by_name(keras_learning_phase_name)
        self.layer_out = self.graph.get_operation_by_name(output_name)
        self.sess = tf.Session(graph=self.graph)
    def predict(self, X):
        preds = self.sess.run(self.layer_out.outputs[0], 
                              {self.layer_in.outputs[0]: X,
                               self.leayer_klp.outputs[0]: 0})
        return preds
    def close(self):
        self.sess.close()


# # Pyaudio Utilities
import pyaudio

def print_pyaudio_devices():
    p = pyaudio.PyAudio()
    count = p.get_device_count()
    for i in range(count):
        dev = p.get_device_info_by_index(i)
        print (i, dev['name'], dev)

