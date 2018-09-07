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
                      preprocessing_function=None):
    # Create Keras ImageDataGenerator
    aug_datagen = ImageDataGenerator(
        featurewise_center=conf.normalize == 'featurewise',
        featurewise_std_normalization=conf.normalize == 'featurewise',
        rotation_range=0,
        width_shift_range=0.4,
        height_shift_range=0.0,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=-1, v_h=1) \
            if preprocessing_function is None else preprocessing_function
    )
    plain_datagen = ImageDataGenerator(
        featurewise_center=aug_datagen.featurewise_center,
        featurewise_std_normalization=aug_datagen.featurewise_std_normalization,
    )
    # Set featurewise normalization mean/std
    if aug_datagen.featurewise_center:
        print(' normalize featurewise')
        aug_datagen.mean, aug_datagen.std = np.mean(_Xtrain), np.std(_Xtrain)
        plain_datagen.mean, plain_datagen.std = aug_datagen.mean, aug_datagen.std
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

def get_cross_valid_fold_balanced(conf, fold, X_train, y_train, idx_train):
    """
    """
    indices = np.array(range(len(X_train)))
    # Cross validation split -> _Xtrain|_ytrain, _Xvalid|_yvalid
    _, _, _, _, train_fold, valid_fold = train_test_split(X_train, y_train, indices,
                                                          test_size=conf.test_size,
                                                          random_state=conf.random_state + fold*10)
    _Xtrain, _ytrain = X_train[train_fold], y_train[train_fold]
    _Xvalid, _yvalid = X_train[valid_fold], y_train[valid_fold]

    # Balance distribution -> _Xtrain|_ytrain (overwritten)
    print_class_balance('Current fold category distribution', _ytrain, conf.labels)
    _Xtrain, _ytrain = balance_class_by_over_sampling(_Xtrain, _ytrain)
    print_class_balance('after balanced', _ytrain, conf.labels)

    return _Xtrain, _ytrain, _Xvalid, _yvalid

def evaluate_model(conf, model, plain_datagen, X, y):
    # Predict
    test_generator = plain_datagen.flow(X, y, batch_size=conf.batch_size, shuffle=False)
    preds = model.predict_generator(test_generator)
    # Calculate accuracy
    targets = np.argmax(y, axis=1)
    results = np.argmax(preds, axis=1)
    acc = np.sum(targets == results) / len(targets)
    print('Accuracy =', acc)
    return acc

def train_model(conf, fold, dataset, model=None, init_weights=None,
                preprocessing_function=None):
    # Split train/valid
    if len(dataset) == 3: # Auto train/valid split
        print('----- Fold #%d ----' % fold)
        X_train, y_train, idx_train = dataset
        # c. Cross validation split & balance # of samples
        _Xtrain, _ytrain, _Xvalid, _yvalid = \
        get_cross_valid_fold_balanced(conf, fold, X_train, y_train, idx_train)
    else: # Or predetermined train/valid split
        Xtrain, ytrain, idxtrain, \
        Xvalid, yvalid, idxvalid = dataset

    # Get generators, steps, callbacks and model
    train_generator, valid_generator, plain_datagen = \
        create_generators(conf, _Xtrain, _ytrain, _Xvalid, _yvalid, preprocessing_function)
    train_steps_per_epoch, valid_steps_per_epoch = \
        get_steps_per_epoch(conf, _Xtrain, _Xvalid)
    callbacks = [
        ModelCheckpoint(str(datapath(conf, conf.best_weight_file)),
                        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir=str(datapath(conf, 'logs/fold_%d' % fold)), write_graph=True)
    ]
    if model is None:
        model = create_model(conf, conf.num_classes, weights=init_weights)
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
