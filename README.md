# Machine Learning Sound Classifier

This is a simple, fast, for real-time use, customizable machine learning sound classifier.

- MobileNetV2, light weight deep learning model, is trained by default.
- Tensorflow graph for runtime prediction for portability.
- Freesound Dataset Kaggle 2018 as default dataset.

## Example: prediction

$ CUDA_VISIBLE_DEVICES= python file_predictor.py sample/fireworks.wav
Using TensorFlow backend.
2018-08-13 11:01:51.739184: E tensorflow/stream_executor/cuda/cuda_driver.cc:397] failed call to cuInit: CUDA_ERROR_NO_DEVICE
2018-08-13 11:01:51.739228: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:157] retrieving CUDA diagnostic information for host: ******
2018-08-13 11:01:51.739238: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:164] hostname: ******
2018-08-13 11:01:51.739281: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:188] libcuda reported version is: 384.130.0
2018-08-13 11:01:51.739308: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:192] kernel reported version is: 384.130.0
2018-08-13 11:01:51.739317: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:299] kernel version seems to match DSO: 384.130.0
Fireworks 31
Fireworks 31
Fireworks 31
Squeak 23
Fireworks 31
Fireworks 31
Fireworks 31
Fireworks 31
Flute 16
Acoustic_guitar 38
Applause 37
Acoustic_guitar 38

## Future works

- Example: Training your dataset

