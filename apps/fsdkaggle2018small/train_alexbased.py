# Train FSDKaggle2018 model
#
import sys
sys.path.append('../..')
from lib_train import *

conf.model = 'alexnet'
conf.logdir = 'logs_alexbased_small'
conf.best_weight_file = 'best_alexbased_small_weight.h5'

# 1. Load Meta data
DATAROOT = Path.home() / '.kaggle/competitions/freesound-audio-tagging'
#Data frame for training dataset
df_train = pd.read_csv(DATAROOT / 'train.csv')
#Plain y_train label
plain_y_train = np.array([conf.label2int[l] for l in df_train.label])

# 2. Preprocess data if it's not ready
def fsdkaggle2018_map_y_train(idx_train, plain_y_train):
    return np.array([plain_y_train[i] for i in idx_train])
def fsdkaggle2018_make_preprocessed_train_data():
    conf.folder.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(conf.X_train):
        XX = mels_build_multiplexed_X(conf, [DATAROOT/'audio_train'/fname for fname in df_train.fname])
        X_train, y_train, X_test, y_test = \
            train_valid_split_multiplexed(conf, XX, plain_y_train, demux=True)
        np.save(conf.X_train, X_train)
        np.save(conf.y_train, y_train)
        np.save(conf.X_test, X_test)
        np.save(conf.y_test, y_test)

fsdkaggle2018_make_preprocessed_train_data()

# 3. Load all dataset & normalize
X_train, y_train = load_audio_datafiles(conf, conf.X_train, conf.y_train, normalize=True)
X_test, y_test = load_audio_datafiles(conf, conf.X_test, conf.y_test, normalize=True)
print('Loaded train:test = {}:{} samples.'.format(len(X_train), len(X_test)))

# 4. Train folds
history, model = train_classifier(conf, fold=0,
                                  dataset=[X_train, y_train, X_test, y_test],
                                  model=None,
                                  init_weights=None, # from scratch
                                  #init_weights='../../model/mobilenetv2_small_fsd2018_41cls.h5'
)

# 5. Evaluate
evaluate_model(conf, model, X_test, y_test)

print('___ training finished ___')
