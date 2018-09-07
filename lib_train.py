# Training functions and definitions
#
# This contains all we need to train.
# Import as follows:
#
# > from lib_train import *
#

from common import *
import os
from random_eraser import get_random_eraser
from mixup_generator import MixupGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import backend as K

# # Dataset Utilities
def mels_populate_samples(org_mels, targ_len, targ_samples):
    """Populates training samples from original full length wave's log mel-spectrogram."""
    org_len = mels_len(org_mels)
    assert org_len >= targ_len
    one_step = np.min([float(targ_len),
                       (org_len - targ_len + 1) / targ_samples])
    generated = []
    for i in range(targ_samples):
        cur = int(one_step * i)
        generated.append(org_mels[:, cur:cur+targ_len])
    return np.array(generated)

def test_mels_populate_samples():
    if True: # Brute force test
        T = 10
        raw = np.zeros((64, T))
        for i in range(T):
            raw[:, i] = i
        raw[0, :] = 0 # placing 0 at the bottom
        for k in range(1, T + 1):
            for i in range(1, 100 + 1):
                try:
                    generated = mels_populate_samples(raw, k, i)
                    if generated.shape[0] != i or generated.shape[2] != k:
                        print('Fail at i={}, k={}, generated.shape={}'.format(i, k, generated.shape))
                except:
                    print('Exception at i={}, k={}'.format(i, k))
                    break
        show_melspectrogram(conf, raw)
        print('Test finished.')
    else: # short test
        k, i = 1, 11
        generated = mels_populate_samples(raw, k, i)
#test_mels_populate_samples()

def mels_build_multiplexed_X(conf, X_files):
    """Builds multiplexed input data (X) from list of files."""
    XX = np.zeros((len(X_files), conf.samples_per_file, conf.dims[0], conf.dims[1]))
    for i, file in enumerate(X_files):
        whole_mels = read_as_melspectrogram(conf, file, trim_long_data=False)
        multiplexed = mels_populate_samples(whole_mels, conf.dims[1], conf.samples_per_file)
        XX[i, :, :, :] = multiplexed
    return XX[..., np.newaxis]

def mels_demux_XX_y(XX, y):
    """De-multiplex data."""
    dims = XX.shape
    X = XX.reshape(dims[0] * dims[1], dims[2], dims[3], dims[4])
    y = np.repeat(y, dims[1], axis=0) # ex. if dims[1]==2) [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
    return X, y

def generate_y_from_pathname(X_files):
    """Generates y data from pathname of files."""
    labeled_y = [Path(f).parent.name for f in X_files]
    labels = sorted(list(set(labeled_y)))
    label2int = {label: i for i, label in enumerate(labels)}
    y = np.array([label2int[a_y] for a_y in labeled_y])
    print('Set labels to config.py; conf.labels =', labels)
    return y, labels, label2int

def train_valid_split_multiplexed(conf, XX, y, demux=True, delta_random_state=0):
    """Splits multiplexed set of data.
    
    XX[sample_no, multiplex_no, ...] to be 3 or more dimentional data vector.
    y[sample_no] can be both one-hot or label index.
    demux==True will output X, False will output raw split XX"""
    assert len(XX) == len(y)
    def split_fold(XX, y, fold_list, demux):
        _XX, _y = XX[fold_list], y[fold_list]
        if demux:
            return mels_demux_XX_y(_XX, _y)
        else:
            return _XX, _y
    # decode y if it is one-hot vector
    _y = y if len(y.shape) == 1 else np.argmax(y, axis=-1)
    print()
    # split train/valid
    train_fold, valid_fold, _, _ = train_test_split(list(range(len(_y))), _y,
                             test_size=conf.test_size,
                             random_state=conf.random_state + delta_random_state)
    X_or_XX_train, y_train = split_fold(XX, y, train_fold, demux)
    X_or_XX_valid, y_valid = split_fold(XX, y, valid_fold, demux)
    # train/valid are ready
    return X_or_XX_train, y_train, X_or_XX_valid, y_valid

def load_dataset(conf, X_or_XX_file, y_file, normalize):
    X_or_XX, y = load_npy(conf, X_or_XX_file), \
                 keras.utils.to_categorical(load_npy(conf, y_file))
    if normalize:
        print(' normalize samplewise')
        if len(X_or_XX.shape) == 5:
            for X in X_or_XX: # it is XX
                samplewise_normalize_audio_X(X)
        else:
            samplewise_normalize_audio_X(X_or_XX) # it is X
    return X_or_XX, y

# # Data Distribution Utilities
def get_class_distribution(y):
    """Calculate number of samples per class."""
    # y_cls can be one of [OH label, index of class, class label name string]
    # convert OH to index of class
    y_cls = [np.argmax(one) for one in y] if len(np.array(y).shape) == 2 else y
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}
    return sample_distribution

def get_class_distribution_list(y, num_classes):
    """Calculate number of samples per class as list"""
    dist = get_class_distribution(y)
    assert(y[0].__class__ != str) # class index or class OH label only
    list_dist = np.zeros((num_classes))
    for i in range(num_classes):
        if i in dist:
            list_dist[i] = dist[i]
    return list_dist

from imblearn.over_sampling import RandomOverSampler
def balance_class_by_over_sampling(X, y):
    Xidx = [[xidx] for xidx in range(len(X))]
    y_cls = [np.argmax(one) for one in y]
    classset = sorted(list(set(y_cls)))
    sample_distribution = [len([one for one in y_cls if one == cur_cls]) for cur_cls in classset]
    nsamples = np.max(sample_distribution)
    flat_ratio = {cls:nsamples for cls in classset}
    Xidx_resampled, y_cls_resampled = RandomOverSampler(ratio=flat_ratio, random_state=42).fit_sample(Xidx, y_cls)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])

def visualize_class_balance(title, y, labels):
    sample_dist_list = get_class_distribution_list(y, len(labels))
    index = range(len(labels))
    fig, ax = plt.subplots(1, 1, figsize = (16, 5))
    ax.bar(index, sample_dist_list)
    ax.set_xlabel('Label')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    fig.show()

def print_class_balance(title, y, labels):
    distributions = get_class_distribution(y)
    dist_dic = {labels[cls]:distributions[cls] for cls in distributions}
    print(title, '=', dist_dic)
    zeroclasses = [label for i, label in enumerate(labels) if i not in distributions.keys()]
    if 0 < len(zeroclasses):
        print(' 0 sample classes:', zeroclasses)

# # Training Functions
from keras.preprocessing.image import ImageDataGenerator

def create_generators(conf, _Xtrain, _ytrain, _Xvalid, _yvalid,
                      image_data_generator=None):
    # Create Keras ImageDataGenerator
    if image_data_generator is None:
        aug_datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.4,
            height_shift_range=0.0,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=1))
        print(' using normal data generator')
    else:
        aug_datagen = image_data_generator
        print(' using special data generator')
    plain_datagen = ImageDataGenerator()
    # Create Generators
    train_generator = MixupGenerator(_Xtrain, _ytrain, 
                                     alpha=1.0, batch_size=conf.batch_size,
                                     datagen=aug_datagen)()
    valid_generator = plain_datagen.flow(_Xvalid, _yvalid,
                                         batch_size=conf.batch_size, shuffle=False)
    return train_generator, valid_generator, plain_datagen

def get_steps_per_epoch(conf, _Xtrain, _Xvalid):
    train_steps_per_epoch = len(_Xtrain) // conf.batch_size
    valid_steps_per_epoch = len(_Xvalid) // conf.batch_size
    return train_steps_per_epoch, valid_steps_per_epoch

def get_cross_valid_fold_balanced(conf, fold, X, y):
    """Gets training set split into train/valid, and balanced."""
    indices = np.array(range(len(X)))
    # Cross validation split -> _Xtrain|_ytrain, _Xvalid|_yvalid
    train_fold, valid_fold, _, _ = train_test_split(indices, y,
                                                    test_size=conf.test_size,
                                                    random_state=conf.random_state + fold*10)
    _Xtrain, _ytrain = X[train_fold], y[train_fold]
    _Xvalid, _yvalid = X[valid_fold], y[valid_fold]

    # Balance distribution -> _Xtrain|_ytrain (overwritten)
    print_class_balance('Current fold category distribution', _ytrain, conf.labels)
    _Xtrain, _ytrain = balance_class_by_over_sampling(_Xtrain, _ytrain)
    print_class_balance('after balanced', _ytrain, conf.labels)

    return _Xtrain, _ytrain, _Xvalid, _yvalid

def calculate_acc_by_preds(y, preds):
    targets = np.argmax(y, axis=1) if len(y.shape) == 2 else y
    results = np.argmax(preds, axis=1)
    acc = np.sum(targets == results) / len(targets)
    return acc

def evaluate_model(conf, model, plain_datagen, X, y):
    # Predict
    test_generator = plain_datagen.flow(X, y, batch_size=conf.batch_size, shuffle=False)
    preds = model.predict_generator(test_generator)
    # Calculate accuracy as is
    acc = calculate_acc_by_preds(y, preds)
    print('Accuracy =', acc)
    # Calculate ensemble accuracy
    if conf.samples_per_file > 1:
        sample_preds_list = np.array([preds[i::conf.samples_per_file]
                                      for i in range(conf.samples_per_file)])
        one_y = y[::conf.samples_per_file]
        ensemble_preds = geometric_mean_preds(sample_preds_list)
        acc = calculate_acc_by_preds(one_y, ensemble_preds)
        print('Ensemble Accuracy =', acc)
    return acc

def train_model(conf, fold, dataset, model=None, init_weights=None,
                image_data_generator=None):
    # Split train/valid
    if len(dataset) == 2: # Auto train/valid split
        print('----- Fold #%d ----' % fold)
        _X_train, _y_train = dataset
        # c. Cross validation split & balance # of samples
        Xtrain, ytrain, Xvalid, yvalid = \
            get_cross_valid_fold_balanced(conf, fold, _X_train, _y_train)
    else: # Or predetermined train/valid split
        Xtrain, ytrain, Xvalid, yvalid = dataset

    # Get generators, steps, callbacks, and model
    train_generator, valid_generator, plain_datagen = \
        create_generators(conf, Xtrain, ytrain, Xvalid, yvalid, image_data_generator)
    train_steps_per_epoch, valid_steps_per_epoch = \
        get_steps_per_epoch(conf, Xtrain, Xvalid)
    callbacks = [
        ModelCheckpoint(str(datapath(conf, conf.best_weight_file)),
                        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir=str(datapath(conf, 'logs/fold_%d' % fold)), write_graph=True)
    ]
    if model is None:
        model = create_model(conf, weights=init_weights)
    # Train model
    history = model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=conf.epochs,
                    validation_data=valid_generator, 
                    validation_steps=valid_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=conf.verbose)
    # Load best weight
    model.load_weights(datapath(conf, conf.best_weight_file))
    return history, model, plain_datagen
