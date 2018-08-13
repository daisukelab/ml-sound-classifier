# Train a model
#

from lib_train import *

print(conf)

#### make preprocessed data if not
make_preprocessed_train_data()

# Load all dataset -> all_(X|y|idx)_train
X_train, y_train, idx_train = \
    loaddata(conf, 'X_train.npy'), \
    keras.utils.to_categorical(loaddata(conf, 'y_train.npy')), \
    loaddata(conf, 'idx_train.npy')
print('Loaded trainset:%d samples.' % (len(X_train)))

# Normalize samplewise if requested
if conf['normalize'] == 'samplewise':
    print(' normalize samplewise')
    samplewise_mean_X(X_train)

# Train folds
work = {'train_acc': [],
        'train_acc_verified': [],
        'history': []}
for fold in range(conf['n_fold']):
    acc, acc_verified, history, model, _ = run_fold(conf, fold,
                [X_train, y_train, idx_train],
                model=None,
                init_best_weights='model/mobilenetv2_fsd2018_41cls.h5', # False, # Set True if you continue from current best one
                eval_only=False)
    work['history'].append(history)
    work['train_acc'].append(acc)
    work['train_acc_verified'].append(acc_verified)

print('___ training finished ___')
