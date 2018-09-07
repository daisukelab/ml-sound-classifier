# Freesound Dataset Kaggle 2018
# Application configurations

from easydict import EasyDict

conf = EasyDict()

# Basic configurations
conf.sampling_rate = 44100
conf.duration = 1
conf.hop_length = 347 # to make time steps 128
conf.fmin = 20
conf.fmax = conf.sampling_rate // 2
conf.n_mels = 128
conf.n_fft = conf.n_mels * 20

# Labels
conf.labels = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
               'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
               'drilling']

# Training configurations
conf.folder = '.'
conf.n_fold = 1
conf.valid_limit = None
conf.random_state = 42
conf.test_size = 0.01
conf.samples_per_file = 5
conf.batch_size = 32
conf.learning_rate = 0.0001
conf.epochs = 500
conf.verbose = 2
conf.best_weight_file = 'best_model_weight.h5'

# Runtime conficurations
conf.rt_process_count = 1
conf.rt_oversamples = 10
conf.pred_ensembles = 10
conf.runtime_model_file = 'model/mobilenetv2_fsd2018_41cls.pb'
