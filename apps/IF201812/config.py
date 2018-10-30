# Interface 2018/12 sound classification example for shaking snack
# Application configurations

from easydict import EasyDict

conf = EasyDict()

# Basic configurations
conf.sampling_rate = 44100
conf.duration = 2
conf.hop_length = 347*2 # to make time steps 128
conf.fmin = 20
conf.fmax = conf.sampling_rate // 2
conf.n_mels = 128
conf.n_fft = conf.n_mels * 20
conf.model = 'mobilenetv2' # 'alexnet'

# Labels
conf.labels = ['babystar', 'bbq', 'corn', 'kappaebi', 'potechi', 'vegetable']

# Training configurations
conf.folder = '.'
conf.n_fold = 1
conf.valid_limit = None
conf.random_state = 42
conf.test_size = 0.2
conf.samples_per_file = 1
conf.batch_size = 32
conf.learning_rate = 0.0001
conf.metric_save_ckpt = 'val_acc'
conf.epochs = 100
conf.verbose = 2
conf.best_weight_file = 'best_model_weight.h5'

# Runtime conficurations
conf.rt_process_count = 1
conf.rt_oversamples = 10
conf.pred_ensembles = 10
conf.runtime_model_file = None # 'model/mobilenetv2_fsd2018_41cls.pb'
