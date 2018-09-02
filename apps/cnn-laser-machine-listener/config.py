# CNN Version Laser Machine Listener
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
conf.labels = ['background', 'cutting_in_focus', 'cutting_not_in_focus',
               'marking', 'sleeping', 'waiting']

# Training configurations
conf.folder = '.'
conf.n_fold = 1
conf.normalize = 'samplewise'
conf.valid_limit = None
conf.random_state = 42
conf.test_size = 0.01
conf.batch_size = 32
conf.learning_rate = 0.0001
conf.epochs = 50
conf.verbose = 2
conf.best_weight_file = 'best_model_weight.h5'

# Runtime conficurations
conf.rt_process_count = 1
conf.rt_oversamples = 10
conf.pred_ensembles = 10
conf.runtime_model_file = 'cnn-model-laser-machine-listener.pb'
