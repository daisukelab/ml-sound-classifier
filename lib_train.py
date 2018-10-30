# Training functions and definitions
#
# This contains all we need to train.
# Import as follows:
#
# > from lib_train import *
#

from common import *
import os
from ext.random_eraser import get_random_eraser
from ext.mixup_generator import MixupGenerator
from ext.balanced_mixup_generator import BalancedMixupGenerator
from ext.clr_callback import CyclicLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import backend as K

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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
    _y = flatten_y_if_onehot(y)
    print()
    # split train/valid
    train_fold, valid_fold, _, _ = train_test_split(list(range(len(_y))), _y,
                             test_size=conf.test_size,
                             random_state=conf.random_state + delta_random_state)
    X_or_XX_train, y_train = split_fold(XX, y, train_fold, demux)
    X_or_XX_valid, y_valid = split_fold(XX, y, valid_fold, demux)
    # train/valid are ready
    return X_or_XX_train, y_train, X_or_XX_valid, y_valid

def load_audio_datafiles(conf, X_or_XX_file, y_file, normalize):
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

def load_datafiles(conf, X_file, y_file=None, normalize=True):
    X = load_npy(conf, X_file)
    if y_file is not None:
        y = keras.utils.to_categorical(load_npy(conf, y_file))
    if normalize:
        print(' normalize samplewise')
        samplewise_normalize_X(X)
    return (X, y) if y_file is not None else X

# # Data Distribution Utilities
def flatten_y_if_onehot(y):
    """De-one-hot y, i.e. [0,1,0,0,...] to 1 for all y."""
    return y if len(np.array(y).shape) == 1 else np.argmax(y, axis = -1)

def get_class_distribution(y):
    """Calculate number of samples per class."""
    # y_cls can be one of [OH label, index of class, class label name string]
    # convert OH to index of class
    y_cls = flatten_y_if_onehot(y)
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

def _balance_class(X, y, min_or_max, sampler_class, random_state):
    """Balance class distribution with sampler_class."""
    y_cls = flatten_y_if_onehot(y)
    distribution = get_class_distribution(y_cls)
    classes = list(distribution.keys())
    counts  = list(distribution.values())
    nsamples = np.max(counts) if min_or_max == 'max' \
          else np.min(counts)
    flat_ratio = {cls:nsamples for cls in classes}
    Xidx = [[xidx] for xidx in range(len(X))]
    sampler_instance = sampler_class(ratio=flat_ratio, random_state=random_state)
    Xidx_resampled, y_cls_resampled = sampler_instance.fit_sample(Xidx, y_cls)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])

def balance_class_by_over_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomOverSampler."""
    return  _balance_class(X, y, 'max', RandomOverSampler, random_state)

def balance_class_by_under_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomUnderSampler."""
    return  _balance_class(X, y, 'min', RandomUnderSampler, random_state)

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
def create_train_generator(conf, _Xtrain, _ytrain, image_data_generator=None):
    # Create Keras ImageDataGenerator
    def print_generator_use(message):
        print(' {}{}'.format(message,
            ', with class balancing' if conf.data_balancing == 'by_generator' else ''))
    if image_data_generator is None:
        aug_datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.4,
            height_shift_range=0.0,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=1))
        print_generator_use('Using normal data generator')
    else:
        aug_datagen = image_data_generator
        print_generator_use('Using Special data generator')
    # Create Generators
    mixup_class = MixupGenerator if conf.data_balancing != 'by_generator' \
                  else BalancedMixupGenerator
    train_generator = mixup_class(_Xtrain, _ytrain, 
                                  alpha=conf.aug_mixup_alpha, batch_size=conf.batch_size,
                                  datagen=aug_datagen)()
    return train_generator

def get_steps_per_epoch(conf, _Xtrain):
    train_steps_per_epoch = (len(_Xtrain) + conf.batch_size - 1) // conf.batch_size
    if conf.steps_per_epoch_limit is not None:
        train_steps_per_epoch = np.clip(train_steps_per_epoch, train_steps_per_epoch,
                                        conf.steps_per_epoch_limit)
    if conf.verbose > 0:
        print(' train_steps_per_epoch, {}'.format(train_steps_per_epoch))
    return train_steps_per_epoch

def balance_dataset(conf, X, y):
    if conf.data_balancing == 'over_sampling' or conf.data_balancing == 'under_sampling':
        print_class_balance(' <Before> Current category distribution', y, conf.labels)
        X, y = balance_class_by_over_sampling(X, y) if conf.data_balancing == 'over_sampling' \
          else balance_class_by_under_sampling(X, y)
        print_class_balance(' <After> Balanced distribution', y, conf.labels)
    else:
        print(' Dataset is not balanced so far, conf.data_balancing =', conf.data_balancing)
    return X, y

def calculate_metrics(conf, y_true, y_pred):
    """Calculate possible metrics."""
    y_true = flatten_y_if_onehot(y_true)
    y_pred = flatten_y_if_onehot(y_pred)
    average = 'weighted' if conf.num_classes > 2 else 'binary'
    f1 = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, recall, precision, accuracy

def skew_preds(y_pred, binary_bias=None):
    _preds = y_pred.copy()
    if binary_bias is not None:
        ps = np.power(_preds[:, 1], binary_bias)
        _preds[:, 1] = ps
        _preds[:, 0] = 1 - ps
        print(' @skew', binary_bias)
    return _preds

def print_metrics(conf, y_true, y_pred, binary_bias=None, title_prefix=''):
    # Add bias if binary_bias is set
    _preds = skew_preds(y_pred, binary_bias)
    # Calculate metrics
    f1, recall, precision, acc = calculate_metrics(conf, y_true, _preds)
    print('{0:s}F1/Recall/Precision/Accuracy = {1:.4f}/{2:.4f}/{3:.4f}/{4:.4f}' \
          .format(title_prefix, f1, recall, precision, acc))

def summarize_metrics_history(metrics_history, show_graph=True):
    """Summarize history of metrics."""
    metrics_history = np.array(metrics_history)
    df=pd.DataFrame({'x': np.arange(1, metrics_history.shape[0]+1),
                     'f1': metrics_history[:, 0],
                     'recall': metrics_history[:, 1],
                     'precision': metrics_history[:, 2],
                     'accuracy': metrics_history[:, 3]})
    print(df[['f1', 'recall', 'precision', 'accuracy']].describe())

    if show_graph:
        plt.plot('x', 'f1', data=df, marker='', color='blue', markersize=2, linewidth=1)
        plt.plot('x', 'recall', data=df, marker='', color='olive', linewidth=1)
        plt.plot('x', 'precision', data=df, marker='o', color='pink', markerfacecolor='red', linewidth=4)
        plt.plot('x', 'accuracy', data=df, marker='', color='black', linewidth=1)
        plt.legend()
        plt.show()

    return df

def evaluate_model(conf, model, X, y):
    # Predict
    preds = model.predict(X)
    # Calculate metrics
    f1, recall, precision, acc = calculate_metrics(conf, y, preds)
    print('F1/Recall/Precision/Accuracy = {0:.4f}/{1:.4f}/{2:.4f}/{3:.4f}' \
          .format(f1, recall, precision, acc))
    # Calculate ensemble accuracy
    if conf.samples_per_file > 1 and conf.eval_ensemble:
        sample_preds_list = np.array([preds[i::conf.samples_per_file]
                                      for i in range(conf.samples_per_file)])
        one_y = y[::conf.samples_per_file]
        ensemble_preds = geometric_mean_preds(sample_preds_list)
        f1, recall, precision, acc = calculate_metrics(conf, one_y, ensemble_preds)
        print('Ensemble F1/Recall/Precision/Accuracy = {0:.4f}/{1:.4f}/{2:.4f}/{3:.4f}' \
              .format(f1, recall, precision, acc))
    return f1, recall, precision, acc

def train_classifier(conf, fold, dataset, model=None, init_weights=None,
                     image_data_generator=None):
    # Test configuration to make sure training properly
    test_conf(conf)
    # Split train/valid
    if len(dataset) == 2: # Auto train/valid split
        print('----- Fold #%d ----' % fold)
        _X_train, _y_train = dataset
        # Cross validation split & balance # of samples
        Xtrain, Xvalid, ytrain, yvalid = \
            train_test_split(_X_train, _y_train,
                             test_size=conf.test_size,
                             random_state=conf.random_state)
    else: # Or predetermined train/valid split
        Xtrain, ytrain, Xvalid, yvalid = dataset

    # Balamce train set
    Xtrain, ytrain = balance_dataset(conf, Xtrain, ytrain)

    # Get generators, steps, callbacks, and model
    train_generator = create_train_generator(conf, Xtrain, ytrain, image_data_generator)
    train_steps_per_epoch = get_steps_per_epoch(conf, Xtrain)
    callbacks = [
        ModelCheckpoint(str(datapath(conf, conf.best_weight_file)),
                        monitor=conf.metric_save_ckpt, mode=conf.metric_save_mode,
                        verbose=1 if conf.verbose > 0 else 0,
                        save_best_only=True, save_weights_only=True),
        CyclicLR(base_lr=conf.learning_rate / 10.0, max_lr=conf.learning_rate,
                 step_size=train_steps_per_epoch, mode='triangular'),
        TensorBoard(log_dir=str(datapath(conf, 'logs/fold_%d' % fold)), write_graph=True)
    ]
    if model is None:
        model = create_model(conf, weights=init_weights)
    # Train model
    history = model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=conf.epochs,
                    validation_data=(Xvalid, yvalid),
                    callbacks=callbacks,
                    verbose=conf.verbose)
    # Load best weight
    model.load_weights(datapath(conf, conf.best_weight_file))
    return history, model

## More visualization

# Thanks to http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """Plot confusion matrix.
    """
    po = np.get_printoptions()
    np.set_printoptions(precision=2)
    
    y_test = flatten_y_if_onehot(y_test)
    y_pred = flatten_y_if_onehot(y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if title is None: title = 'Normalized confusion matrix'
    else:
        if title is None: title = 'Confusion matrix (not normalized)'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    np.set_printoptions(**po)