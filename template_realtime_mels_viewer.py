#-*-coding:utf-8-*-
#!/usr/bin/python
#
# This is template python application for printing log mel-spectrogram as text.
# You can edit for your purpose.
#
from common import *
import pyaudio
import sys
import time
import array
import numpy as np
import queue
import argparse

parser = argparse.ArgumentParser(description='Sample log mel-spectrogram viewer')
parser.add_argument('--input-file', '-f', default='', type=str,
                    help='If set, process this audio file.')
args = parser.parse_args()

conf={}
conf['sampling_rate'] = 44100
conf['duration'] = 1
conf['hop_length'] = conf['sampling_rate'] // 10 # 347 -> to make time steps 128
conf['fmin'] = 20
conf['fmax'] = conf['sampling_rate'] // 2
conf['n_mels'] = 64
conf['n_fft'] = conf['n_mels'] * 20
conf['rt_process_count'] = 1
conf['rt_oversamples'] = 10
conf['rt_chunk_samples'] = conf['sampling_rate'] // conf['rt_oversamples']
conf['audio_split'] = 'dont_crop'
auto_complete_conf(conf)

mels_onestep_samples = conf['rt_chunk_samples'] * conf['rt_process_count']
mels_convert_samples = conf['samples'] + mels_onestep_samples

raw_frames = queue.Queue(maxsize=100)
def callback(in_data, frame_count, time_info, status):
    wave = array.array('h', in_data)
    raw_frames.put(wave, True)
    return (None, pyaudio.paContinue)

def level2char(level, amin=0, amax=40):
    chrs = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588']
    level = np.clip(level, amin, amax)
    index = int((level - amin - 1e-3) / (amax - amin) * len(chrs))
    return chrs[index]

raw_audio_buffer = []
def main_process():
    global raw_audio_buffer
    while not raw_frames.empty():
        raw_audio_buffer.extend(raw_frames.get())
        if len(raw_audio_buffer) >= mels_convert_samples: break
    if len(raw_audio_buffer) < mels_convert_samples: return

    audio_to_convert = np.array(raw_audio_buffer[:mels_convert_samples]) / 32767.0
    raw_audio_buffer = raw_audio_buffer[mels_onestep_samples:]
    mels = audio_to_melspectrogram(conf, audio_to_convert)
    for i in range(mels.shape[1]):
    	#print(' '.join(['%2d' % int(x) for x in mels[:,0]]))
        print(''.join([level2char(x) for x in mels[:,i]]))

def process_file(filename):
    # Feed audio data as if it was recorded in realtime
    audio = read_audio(conf, filename) * 32767
    while len(audio) > conf['rt_chunk_samples']:
        raw_frames.put(audio[:conf['rt_chunk_samples']])
        audio = audio[conf['rt_chunk_samples']:]
        main_process()

if __name__ == '__main__':
    if args.input_file != '':
        process_file(args.input_file)
        exit(0)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=conf['sampling_rate'],
                input=True,
                #input_device_index=1,
                frames_per_buffer=conf['rt_chunk_samples'],
                start=False,
                stream_callback=callback # uncomment for non_blocking
            )

    stream.start_stream()
    while stream.is_active():
        main_process()
        time.sleep(0.001)
    stream.stop_stream()
    stream.close()

    audio.terminate()
