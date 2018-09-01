from realtime_predictor import *

emoji = {'Writing': '\U0001F4DD ', 'Scissors': '\u2701 ',
         'Computer_keyboard': '\u2328 '}

def on_predicted_deskwork(ensembled_pred):
    result = np.argmax(ensembled_pred)
    label = conf.labels[result]
    if label in ['Writing', 'Scissors', 'Computer_keyboard']:
        p = ensembled_pred[result]
        level = int(p*10) + 1
        print(emoji[label] * level, label, p)

if __name__ == '__main__':
    model = get_model(args.model_pb_graph)
    # file mode
    if args.input_file != '':
        process_file(model, args.input_file, on_predicted_deskwork)
        my_exit(model)
    # device list display mode
    if args.input < 0:
        print_pyaudio_devices()
        my_exit(model)
    # normal: realtime mode
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=conf.sampling_rate,
                input=True,
                input_device_index=args.input,
                frames_per_buffer=conf.rt_chunk_samples,
                start=False,
                stream_callback=callback # uncomment for non_blocking
            )
    # main loop
    stream.start_stream()
    while stream.is_active():
        main_process(model, on_predicted_deskwork)
        time.sleep(0.001)
    stream.stop_stream()
    stream.close()
    # finish
    audio.terminate()
    my_exit(model)


