# Train FSDKaggle2018 model
#
import sys
sys.path.append('../..')
from lib_train import *

# Load all dataset & normalize
X_train, y_train = load_dataset(conf, 'X_train.npy', 'y_train.npy', normalize=True)
X_test, y_test = load_dataset(conf, 'X_test.npy', 'y_test.npy', normalize=True)
print('Loaded train:test = {}:{} samples.'.format(len(X_train), len(X_test)))

# Train folds
history, model, plain_datagen = train_classifier(conf, fold=0,
                                                 dataset=[X_train, y_train],
                                                 model=None,
                                                 #init_weights=None, # from scratch
                                                 init_weights='../../model/mobilenetv2_fsd2018_41cls.h5')
acc = evaluate_model(conf, model, plain_datagen, X_test, y_test)

print('___ training finished ___')
