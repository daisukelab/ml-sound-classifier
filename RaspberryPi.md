# Running on Raspberry Pi

This implementation uses full of CPU, not specialized to use GPU on it. You will need Raspberry Pi 3.

## Install modeles

Followings are my installation log, you might be able to install on fresh Raspbian image.

```sh
sudo apt-get install python3-numpy
sudo apt-get install python3-scipy
sudo apt-get install python3-pandas
sudo apt-get install python3-h5py
sudo apt install libatlas-base-dev
sudo apt-get install python3-pyaudio
sudo pip3 install resampy==0.1.5 librosa==0.5.1
pip3 install tensorflow
pip3 install keras
pip3 install easydict
git clone https://github.com/daisukelab/ml-sound-classifier.git
cd ml-sound-classifier/ext
./download.sh
```

Tensorflow 1.9 will be installed as of now.

LibROSA has to be the version above, thanks to [the web article (in Japanese) "Raspbian Stretch with Desktopにlibrosaをインストールする@Qiita"](https://qiita.com/mayfair/items/92874e69ba63378f6280).

You can confirm by running off-line inference as follows:

```sh
cd ml-sound-classifier
python3 premitive_file_predictor.py sample/fireworks.wav
```

** This uses MobileNetV2 model, and it's not fast enough.

## Run realtime inference

```sh
cd ml-sound-classifier/rpi
python3 ../realtime_predictor.py -i 2
```

The `rpi` folder has appropriate `config.py` for using `alexbased_small_fsd2018_41cls.h5` pre-trained model.

- 0.5s interval to output results; `conf.rt_process_count / conf.rt_oversamples = 5 / 10 = 0.5`
- It still calculates ensemble of 10 predictions.

