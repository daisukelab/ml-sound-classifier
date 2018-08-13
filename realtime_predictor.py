#-*-coding:utf-8-*-
#!/usr/bin/python
#
# Run sound classifier in realtime.
#
from common import *

import pyaudio
import wave
import sys
import time
import array
import numpy as np
import queue
from collections import deque
import argparse

parser = argparse.ArgumentParser(description='Run sound classifier')
parser.add_argument('--input', '-i', default='0', type=int,
                    help='Audio input device index. Set -1 to list devices')
parser.add_argument('--input-file', '-f', default='', type=str,
                    help='If set, predict this audio file.')
#parser.add_argument('--save_file', default='recorded.wav', type=str,
#                    help='File to save samples captured while running.')
parser.add_argument('--model-pb-graph', '-m', default='model/sample_model.pb', type=str,
                    help='Feed model you want to run.')
args = parser.parse_args()

conf={}
conf['sampling_rate'] = 44100
conf['duration'] = 1
conf['hop_length'] = 347 # to make time steps 128
conf['fmin'] = 20
conf['fmax'] = conf['sampling_rate'] // 2
conf['n_mels'] = 128
conf['n_fft'] = conf['n_mels'] * 20
conf['rt_process_count'] = 1
conf['rt_oversamples'] = 10
conf['rt_chunk_samples'] = conf['sampling_rate'] // conf['rt_oversamples']
conf['pred_ensembles'] = 10
conf['rt_power_threshold'] = -10000
conf['audio_split'] = 'dont_crop'
auto_complete_conf(conf)

mels_onestep_samples = conf['rt_chunk_samples'] * conf['rt_process_count']
mels_convert_samples = conf['samples'] + mels_onestep_samples

# # Capture & pridiction jobs
raw_frames = queue.Queue(maxsize=100)
def callback(in_data, frame_count, time_info, status):
    wave = array.array('h', in_data)
    raw_frames.put(wave, True)
    return (None, pyaudio.paContinue)

raw_audio_buffer = []
pred_queue, power_queue = [deque(maxlen=conf['pred_ensembles']) for _ in range(2)]
def main_process(model, pred_mean_fn=geometric_mean_preds):
    # Pool audio data
    global raw_audio_buffer
    while not raw_frames.empty():
        raw_audio_buffer.extend(raw_frames.get())
        if len(raw_audio_buffer) >= mels_convert_samples: break
    if len(raw_audio_buffer) < mels_convert_samples: return
    # Convert to log mel-spectrogram
    audio_to_convert = np.array(raw_audio_buffer[:mels_convert_samples]) / 32767
    raw_audio_buffer = raw_audio_buffer[mels_onestep_samples:]
    mels = audio_to_melspectrogram(conf, audio_to_convert)
    X = []
    for i in range(conf['rt_process_count']):
        cur = int(i * conf['dims'][1] / conf['rt_oversamples'])
        X.append(mels[:, cur:cur+conf['dims'][1], np.newaxis])
    X = np.array(X)
    raw_powers = [np.sum(one_x) for one_x in X]
    samplewise_mean_X(X)
    # Predict, ensemble
    raw_preds = model.predict(X)
    for raw_pred, raw_power in zip(raw_preds, raw_powers):
        #if raw_power < conf['rt_power_threshold']:
        #    pred_queue.clear()
        #    power_queue.clear()
        #    continue
        pred_queue.append(raw_pred)
        power_queue.append(raw_power)
        ensembled_pred = pred_mean_fn(np.array([pred for pred in pred_queue]))
        mean_power = np.mean(power_queue)
        result = np.argmax(ensembled_pred)
        print(labels[result], result, ensembled_pred[result], mean_power, raw_frames.qsize())

# # Main controller
def process_file(model, filename):
    # Feed audio data as if it was recorded in realtime
    audio = read_audio(conf, filename) * 32767
    while len(audio) > conf['rt_chunk_samples']:
        raw_frames.put(audio[:conf['rt_chunk_samples']])
        audio = audio[conf['rt_chunk_samples']:]
        main_process(model)

def my_exit(model):
    model.close()
    exit(0)

if __name__ == '__main__':
    model = KerasTFGraph(args.model_pb_graph,
        input_name='import/input_1',
        keras_learning_phase_name='import/bn_Conv1/keras_learning_phase',
        output_name='import/output0')

    if args.input_file != '':
        process_file(model, args.input_file)
        my_exit(model)

    if args.input < 0:
        print_pyaudio_devices()
        my_exit(model)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=conf['sampling_rate'],
                input=True,
                input_device_index=args.input,
                frames_per_buffer=conf['rt_chunk_samples'],
                start=False,
                stream_callback=callback # uncomment for non_blocking
            )

    # main loop
    i = 0
    stream.start_stream()
    while stream.is_active():
        main_process(model)
        i += 1
        time.sleep(0.001)
    stream.stop_stream()
    stream.close()
    # finish
    audio.terminate()
    my_exit(model)


