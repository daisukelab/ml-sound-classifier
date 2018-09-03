# Data preprocessing class
#
# This helps preprocessing data for preparing for training.
# Import as follows:
#
# > from lib_data_preprocess import *
#

from lib_train import *

import random
from sklearn.model_selection import train_test_split

def mels_len(mels): return mels.shape[1]
def conf_mels_len(conf): return conf['dims'][1]

class AudioMelsExclusiveSplit:
    """Exclusive splittr for Audio log mel-spectrogram.
    
    Input raw_list is supposed to be a list of raw samples.
    Each raw sample is suposed to be a np.array((n_mels, length of sample)).
    Raw samples can have different lengths."""

    def __init__(self, conf, raw_list):
        self.conf = conf
        self.raw_list = raw_list

    def _train_valid_split(self):
        if len(self.raw_list) <= 1:
            # We really don't like... but no way sourcing from the same distribution
            valid_len = int(mels_len(self.raw_list[0]) * conf['test_size'])
            self.mels_valid = [self.raw_list[0][:, :valid_len]]
            self.mels_train = [self.raw_list[0][:, valid_len:]]
        else:
            self.mels_train = self.raw_list[:-1]
            self.mels_valid = self.raw_list[-1:]
        print('train_valid_split', [a.shape for a in self.mels_train],
             [a.shape for a in self.mels_valid])

    def _get_random_sample_mels(self, mels_set):
        m = random.randint(0, len(mels_set) - 1)
        pos_max = np.max([0, mels_len(mels_set[m]) - conf_mels_len(self.conf) - 1])
        pos = random.randint(0, pos_max)
        #print('randomly chosen:', m, pos, 'for mels.shape', mels_set[m].shape)
        cut_mels = mels_set[m][:, pos:pos+conf_mels_len(self.conf)]
        if cut_mels.shape[0] == 0:
            print(mels_set[m].shape, m, pos, mels_len(mels_set[m]), 
                  pos+conf_mels_len(self.conf))
        return cut_mels
    
    def get_train_valid_set(self, N):
        """Get train/valid samples with special care not to leak each other.
        
        Raw Samples are split into train/valid first,
        then train/valid samples are cut from both raw samples exclusively."""
        self._train_valid_split()
        n_valid = int(N*conf['test_size'])
        n_train = N - n_valid
        train_set = np.array([self._get_random_sample_mels(self.mels_train) \
                              for _ in range(n_train)])
        valid_set = np.array([self._get_random_sample_mels(self.mels_valid) \
                              for _ in range(n_valid)])
        return train_set, valid_set

class AudioDataPreprocessor:
    def __init__(self, conf):
        self.conf = conf
    def f(self, file):
        return str(self.conf.folder/file)
    def convert_by_exclusive_split(self, file_list, nsample_per_class=100):
        """Convert audio files as log mel-spectrogram data, split train/valid,
        then random cut to augment from raw data to populate up to nsample_per_class."""
        labels = sorted(list(set([Path(f).parent.name for f in file_list])))
        label2int = {label: i for i, label in enumerate(labels)}
        class_files = {label: [f for f in file_list if f.parent.name == label]
                        for label in labels}
        X_train, X_valid = [], []
        y_train, y_valid = [], []
        for label in labels:
            print('[%s]' % label, 'has', len(class_files[label]), 'files.')
            this_class_mels_set = [read_as_melspectrogram(conf, f) \
                                   for f in class_files[label]]
            splitter = AudioMelsExclusiveSplit(conf, this_class_mels_set)
            train, valid = splitter.get_train_valid_set(nsample_per_class)
            X_train.extend(train)
            X_valid.extend(valid)
            y_train.extend([label2int[label]] * len(train))
            y_valid.extend([label2int[label]] * len(valid))
        self.X_train = np.array(X_train)[..., np.newaxis]
        self.X_valid = np.array(X_valid)[..., np.newaxis]
        self.y_train, self.y_valid = np.array(y_train), np.array(y_valid)
        print('Set labels to config.py', labels)
    def write_all(self):
        """Write all preprocessed data to files."""
        self.conf.folder.mkdir(parents=True, exist_ok=True)
        np.save(self.f('X_train.npy'), self.X_train)
        np.save(self.f('X_valid.npy'), self.X_valid)
        np.save(self.f('y_train.npy'), self.y_train)
        np.save(self.f('y_valid.npy'), self.y_valid)
        np.save(self.f('idx_train.npy'), list(range(len(self.X_train))))
        np.save(self.f('idx_valid.npy'), list(range(len(self.X_valid))))
        print('Train set {} samples.'.format(len(self.X_train)))
        print('Valid set {} samples.'.format(len(self.X_valid)))
        print('Wrote preprocessed training data.')
    def resuffle_train_valid(self):
        """Merge train/valid data once, then split again.
        
        Note: This might cause data leak, 
              check if resuffling really works with your application."""
        X = np.r_[self.X_train, self.X_valid]
        y = np.r_[self.y_train, self.y_valid]
        self.X_train, self.X_valid, self.y_train, self.y_valid = \
            train_test_split(X, y, test_size=conf.test_size, random_state=conf.random_state)
        print('Re-shuffled train/valid split with test size = {}.'.format(conf.test_size))
