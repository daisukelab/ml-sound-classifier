# Train FSDKaggle2018 model
#
import sys
sys.path.append('../..')
from lib_train import *

print(conf)

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
    if not os.path.exists(datapath(conf, 'X_train.npy')):
        X_train, idx_train = convert_X(df_train.fname, conf, DATAROOT / 'audio_train')
        np.save(datapath(conf, 'X_train.npy'), X_train)
        np.save(datapath(conf, 'idx_train.npy'), idx_train)
        y_train = fsdkaggle2018_map_y_train(idx_train, plain_y_train)
        np.save(datapath(conf, 'y_train.npy'), y_train)

fsdkaggle2018_make_preprocessed_train_data()

# 3. Load all dataset -> all_(X|y|idx)_train
X_train, y_train, idx_train = \
    loaddata(conf, 'X_train.npy'), \
    keras.utils.to_categorical(loaddata(conf, 'y_train.npy')), \
    loaddata(conf, 'idx_train.npy')
print('Loaded trainset:%d samples.' % (len(X_train)))

# 4. Normalize samplewise if requested
if conf['normalize'] == 'samplewise':
    print(' normalize samplewise')
    samplewise_normalize_audio_X(X_train)

# 5. Train folds
history, model, plain_datagen = train_model(conf, fold=0,
                                            dataset=[X_train, y_train, idx_train],
                                            model=None,
                                            #init_weights=None, # from scratch
                                            init_weights='../../model/mobilenetv2_fsd2018_41cls.h5')
acc = evaluate_model(conf, model, plain_datagen, X_valid, y_valid)

print('___ training finished ___')
