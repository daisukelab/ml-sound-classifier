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

# # Additional Training Configurations

conf['folder'] = Path('folder_train')
conf['n_fold'] = 1
conf['normalize'] = 'samplewise'
conf['valid_limit'] = None
conf['random_state'] = 42
conf['test_size'] = 0.01
conf['batch_size'] = 32
conf['learning_rate'] = 0.0001
conf['epochs'] = 500
conf['batch_size'] = 32
conf['verbose'] = 2

DATAROOT = Path.home() / '.kaggle/competitions/freesound-audio-tagging'
# Data frame for train dataset
df_train = pd.read_csv(DATAROOT / 'train.csv')
# Train data sample index of manually verified ones
train_verified_idx = np.array(df_train[df_train.manually_verified == 1].index)
# Plain y_train label
plain_y_train = np.array([label2int[label] for label in df_train.label])

# # Training Dataset Utilities
# Data utilities
def datapath(conf, filename):
    return conf['folder'] / filename

def loaddata(conf, filename):
    return np.load(conf['folder'] / filename)

def make_preprocessed_train_data():
    conf['folder'].mkdir(parents=True, exist_ok=True)
    if not os.path.exists(datapath(conf, 'X_train.npy')):
        X_train, idx_train = convert_X(df_train, conf, DATAROOT / 'audio_train')
        y_train = convert_y_train(idx_train, plain_y_train)
        np.save(datapath(conf, 'X_train.npy'), X_train)
        np.save(datapath(conf, 'y_train.npy'), y_train)
        np.save(datapath(conf, 'idx_train.npy'), idx_train)

def get_class_distribution(y):
    # y_cls can be one of [OH label, index of class, class label name]
    # convert OH to index of class
    y_cls = [np.argmax(one) for one in y] if len(np.array(y).shape) == 2 else y
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}
    return sample_distribution

def get_class_distribution_list(y, num_classes):
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

def create_generators(conf, _Xtrain, _ytrain, _Xvalid, _yvalid):
    # Create Keras ImageDataGenerator
    aug_datagen = ImageDataGenerator(
        featurewise_center=conf['normalize'] == 'featurewise',
        featurewise_std_normalization=conf['normalize'] == 'featurewise',
        rotation_range=0,
        width_shift_range=0.4,
        height_shift_range=0.0,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=-1, v_h=1)
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
                                     alpha=1.0, batch_size=conf['batch_size'], datagen=aug_datagen)()
    valid_generator = plain_datagen.flow(_Xvalid, _yvalid,
                                         batch_size=conf['batch_size'], shuffle=False)
    return train_generator, valid_generator, plain_datagen

def get_steps_per_epoch(conf, _Xtrain, _Xvalid):
    train_steps_per_epoch = len(_Xtrain) // conf['batch_size']
    valid_steps_per_epoch = len(_Xvalid) // conf['batch_size']
    return train_steps_per_epoch, valid_steps_per_epoch

##
from sklearn.model_selection import train_test_split
def get_cross_valid_fold_balanced(conf, fold, X_train, y_train, idx_train):
    indices = np.array(range(len(X_train)))
    # Cross validation split -> _Xtrain|_ytrain, _Xvalid|_yvalid
    _, _, _, _, train_fold, valid_fold = train_test_split(X_train, y_train, indices,
                                                          test_size=conf['test_size'],
                                                          random_state=conf['random_state'] + fold*10)
    _Xtrain, _ytrain = X_train[train_fold], y_train[train_fold]

    # Validation set to filter non-verified samples if requested
    if conf['valid_limit'] == 'manually_verified_only':
        filtered = [idx for idx in valid_fold if idx_train[idx] in train_verified_idx]
        print(' valid set is filtered to verified samples only, %d -> %d' % (len(valid_fold), len(filtered)))
        valid_fold = filtered
    _Xvalid, _yvalid = X_train[valid_fold], y_train[valid_fold]

    # Balance distribution -> _Xtrain|_ytrain (overwritten)
    print_class_balance('Current fold category distribution', _ytrain, labels)
    _Xtrain, _ytrain = balance_class_by_over_sampling(_Xtrain, _ytrain)
    print_class_balance('after balanced', _ytrain, labels)

    return _Xtrain, _ytrain, _Xvalid, _yvalid

from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import backend as K
def train_model(conf, fold, model, train_generator, valid_generator,
                train_steps_per_epoch, valid_steps_per_epoch,
                init_best_weights=False, this_epochs=None):
    callbacks = [
        ModelCheckpoint(str(datapath(conf, 'best_%d.h5' % fold)),
                        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir=str(datapath(conf, 'logs%s/fold_%d' % (conf['folder'], fold))), write_graph=True)
    ]
    # Create model
    if model is None:
        model = create_model(conf, num_classes)
        #if fold == 0:
        #    model.summary()
        # Load weights
        weight_filename = str(init_best_weights) # for when file name was set
        if weight_filename == 'True': weight_filename = str(datapath(conf, 'best_%d.h5' % fold))
        if weight_filename is not 'False':
            print(' Initializing model with last best weights:', weight_filename)
            model.load_weights(weight_filename)
    # Train model
    history = model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=conf['epochs'] if this_epochs is None else this_epochs,
                    validation_data=valid_generator, 
                    validation_steps=valid_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=conf['verbose'])
    return model, history

##
def geometric_mean_preds(_preds):
    preds = _preds.copy()
    for i in range(1, preds.shape[0]):
        preds[0] = np.multiply(preds[0], preds[i])
    return preds[0]

def get_unified_preds(preds, pred_idx, N):
    mean_results = []
    for idx in range(N):
        this_preds = preds[np.where(pred_idx == idx)]
        if len(this_preds) <= 0:
            print(' no result: %d' % idx)
            mean_results.append(np.ones((preds.shape[1],)))
        else:
            mean_results.append(geometric_mean_preds(this_preds))
    return np.array(mean_results)

def evaluate_pred_acc(y, uni_preds, idx_map, N):
    uni_y = []
    for idx in range(N):
        uni_y.append(y[np.where(idx_map == idx)[0]][0])
    uni_y = np.array(uni_y)

    refs = np.argmax(uni_y, axis=1)
    results = np.argmax(uni_preds, axis=1)
    acc = np.sum(refs == results) / len(refs)
    n_verified = len(refs[train_verified_idx])
    acc_verified = np.sum(refs[train_verified_idx] == results[train_verified_idx]) / n_verified
    return acc, acc_verified

def evaluate_fold(conf, fold, filenametmpl, model, plain_datagen, X, idx_map, y=None, verified_idx=None):
    # predict
    _y = keras.utils.to_categorical(np.ones((len(X)))) if y is None else y
    test_generator = plain_datagen.flow(X, _y, batch_size=conf['batch_size'], shuffle=False)
    preds = model.predict_generator(test_generator)
    preds = get_unified_preds(preds, idx_map, np.max(idx_map) + 1)
    # save & return acc
    np.save(datapath(conf, filenametmpl % fold), preds)
    # evaluate 
    if y is not None:
        return evaluate_pred_acc(y, preds, idx_map, len(plain_y_train))
    return None, None

def run_fold(conf, fold, dataset, model=None, init_best_weights=False, eval_only=False):
    X_train, y_train, idx_train = dataset
    print('----- Fold#%d ----' % fold)
    # c. Cross validation split & balance # of samples
    _Xtrain, _ytrain, _Xvalid, _yvalid = \
        get_cross_valid_fold_balanced(conf, fold, X_train, y_train, idx_train)

    # d. Train model
    train_generator, valid_generator, plain_datagen = \
        create_generators(conf, _Xtrain, _ytrain, _Xvalid, _yvalid)
    train_steps_per_epoch, valid_steps_per_epoch = \
        get_steps_per_epoch(conf, _Xtrain, _Xvalid)
    model, history = train_model(conf, fold, model, train_generator, valid_generator,
                                train_steps_per_epoch, valid_steps_per_epoch,
                                 init_best_weights=init_best_weights,
                                this_epochs=0 if eval_only else None)

    # e. Evaluate with all train sample
    model.load_weights(datapath(conf, 'best_%d.h5' % fold))
    acc, acc_v = evaluate_fold(conf, fold, 'train_predictions_%d.npy', model, plain_datagen,
                               X_train, idx_train, y_train, train_verified_idx)

    print('Trainset accuracy =', acc, '(tested all over the original training set)')
    print('Verified samples accuracy =', acc_v, '(tested over manually verified samples only)')
    return acc, acc_v, history, model, plain_datagen



