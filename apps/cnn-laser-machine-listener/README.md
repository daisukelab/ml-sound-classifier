## Running predictor

```sh
python ../../realtime_predictor.py
```

This will load .pb model file specified by `conf.runtime_model_file`, and start listening to the audio input.

or specifically set .pb file.

```sh
python ../../realtime_predictor.py -pb cnn-model-laser-machine-listener.pb
```

Following another example below will not use CUDA and predict from a file.

```sh
CUDA_VISIBLE_DEVICES= python ../../realtime_predictor.py -f laser-machine-listener/data/cutting_not_in_focus/paper_cutting_not_in_focus.wav
```

## Training model

See [EXAMPLE_APPS.md](../../EXAMPLE_APPS.md).

