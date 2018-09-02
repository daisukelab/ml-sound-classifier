# Freesound Dataset Kaggle 2018
# Application configurations

from easydict import EasyDict

conf = EasyDict()

# Basic configurations
conf.sampling_rate = 16000
conf.duration = 1
conf.hop_length = 253 # to make time steps 64
conf.fmin = 20
conf.fmax = conf.sampling_rate // 2
conf.n_mels = 64
conf.n_fft = conf.n_mels * 20
conf.audio_split = 'dont_crop'

# Labels
conf.labels = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
       'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard',
       'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing',
       'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime',
       'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica', 'Gong',
       'Microwave_oven', 'Burping_or_eructation', 'Double_bass', 'Shatter',
       'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano', 'Meow',
       'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
       'Violin_or_fiddle', 'Finger_snapping']

# Training Configurations
conf.folder = '.'
conf.n_fold = 1
conf.normalize = 'samplewise'
conf.valid_limit = None
conf.random_state = 42
conf.test_size = 0.01
conf.batch_size = 32
conf.learning_rate = 0.0001
conf.epochs = 500
conf.verbose = 2
conf.best_weight_file = 'best_model_weight.h5'

# Runtime conficurations
conf.rt_process_count = 1
conf.rt_oversamples = 10
conf.pred_ensembles = 10
conf.runtime_model_file = 'please make yourself' ### NOT PROVIDED
