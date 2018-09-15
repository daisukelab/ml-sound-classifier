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
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd()))

# # Configration
def is_handling_audio(conf):
    return 'sampling_rate' in conf

def test_conf(conf):
    if conf.model not in ['mobilenetv2', 'alexnet']:
        raise Exception('conf.model not recognized: {}'.format(conf.model))
    if conf.data_balancing not in ['over_sampling', 'under_sampling',
                                        'by_generator', 'dont_balance']:
        raise Exception('conf.data_balancing not recognized: {}'.format(conf.data_balancing))

def auto_complete_conf(conf):
    if 'folder' in conf:
        conf.folder = Path(conf.folder)
    conf.label2int = {l:i for i, l in enumerate(conf.labels)}
    conf.num_classes = len(conf.labels)
    # audio auto configurations
    if is_handling_audio(conf):
        conf.samples = conf.sampling_rate * conf.duration
        conf.rt_chunk_samples = conf.sampling_rate // conf.rt_oversamples
        conf.mels_onestep_samples = conf.rt_chunk_samples * conf.rt_process_count
        conf.mels_convert_samples = conf.samples + conf.mels_onestep_samples
        conf.dims = (conf.n_mels, 1 + int(np.floor(conf.samples/conf.hop_length)), 1)
    # optional configurations
    if 'model' not in conf:
        conf.model = 'mobilenetv2'
    if 'metric_save_ckpt' not in conf:
        conf.metric_save_ckpt = 'val_loss'
    if 'metric_save_mode' not in conf:
        conf.metric_save_mode='auto'
    if 'logdir' not in conf:
        conf.logdir = 'logs'
    if 'data_balancing' not in conf:
        conf.data_balancing = 'over_sampling'
    if 'X_train' not in conf:
        conf.X_train = 'X_train.npy'
        conf.y_train = 'y_train.npy'
        conf.X_test  = 'X_test.npy'
        conf.y_test  = 'y_test.npy'
    if 'steps_per_epoch_limit' not in conf:
        conf.steps_per_epoch_limit = None
    if 'samples_per_file' not in conf:
        conf.samples_per_file = 1

from config import *
auto_complete_conf(conf)
print(conf)

# # Data utilities
def load_labels(conf):
    conf.labels = load_npy(conf, 'labels.npy')
    auto_complete_conf(conf)
    print('Labels are', conf.labels)

def datapath(conf, filename):
    return conf.folder / filename

def load_npy(conf, filename):
    return np.load(conf.folder / filename)

# # Model
if is_handling_audio(conf):
    from model import create_model, freeze_model_layers

# # Audio Utilities
import librosa
import librosa.display

def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
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

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    import IPython
    import matplotlib
    from sklearn.model_selection import StratifiedKFold
    matplotlib.style.use('ggplot')

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
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

def samplewise_normalize_X(X):
    for i in range(len(X)):
        X[i] -= np.min(X[i])
        X[i] /= (np.max(np.abs(X[i])) + 1e-07) # same as K.epsilon()

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

def mels_len(mels):
    """Gets lenfth of log mel-spectrogram."""
    return mels.shape[1]

def audio_sample_to_X(conf, wave):
    mels = audio_to_melspectrogram(conf, wave)
    X = []
    for s in range(0, mels_len(mels) // conf.dims[1]):
        cur = s * conf.dims[1]
        X.append(mels[:, cur:cur + conf.dims[1]][..., np.newaxis])
    X = np.array(X)
    samplewise_normalize_audio_X(X)
    return X

def load_sample_as_X(conf, filename, trim_long_data):
    wave = read_audio(conf, filename, trim_long_data)
    return audio_sample_to_X(conf, wave)

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
if is_handling_audio(conf):
    import pyaudio

def print_pyaudio_devices():
    p = pyaudio.PyAudio()
    count = p.get_device_count()
    for i in range(count):
        dev = p.get_device_info_by_index(i)
        print (i, dev['name'], dev)

# # Test Utilities
def recursive_test(a, b, fn):
    """Greedy test every single corresponding contents between a & b recursively."""
    if isinstance(a, (list, set, tuple, np.ndarray)):
        results = np.array([test_equal(aa, bb) for aa, bb in zip(a, b)])
        #print(results) # for debug
        return np.all(results == 1)
    else:
        return 1 if np.all(fn(a, b)) else 0

def test_equal(a, b):
    """Exhaustively test if a equals b"""
    return recursive_test(a, b, lambda a, b: a == b)

def test_not_equal(a, b):
    """Exhaustively test if a != b"""
    return not test_equal(a, b)

