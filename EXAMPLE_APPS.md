# Example Applications

`apps` folder has some example applications which explains entire process to work with your dataset.

## FSDKaggle2018

This trains a model for Freesound Dataset Kaggle 2018 dataset.

- `mobilenetv2_fsd2018_41cls.h5` is a trained model for 500 epochs, almost competitive even in the Kaggle competition.
- It has useful representation for using as pretrained model for other tasks.
- 44.1 kHz, 1 second duration, 128 n_mels and 128 time hops.

Running this will train model:

```Python
cd apps/fsdkaggle2018
python train.py
```

## FSDKaggle2018small

This trains a model for handing smaller audio data. Model is not small but audio, FS is 16 kHz for example.　This sample is less computationally expensive for audio processing.

- `mobilenetv2_small_fsd2018_41cls.h5` is the model created by this.
- 16 kHz, 1 second duration, 64 n_mels and 64 time hops.

Running this will train model:

```Python
cd apps/fsdkaggle2018small
python train_this.py
```

## CNN Laser Machine Listener

This is a experimental application example to [github/Laser Machine Listener](https://github.com/kotobuki/laser-machine-listener).
Classification problem of sounds in hardware laboratory.

Originally simple NN was applied successfully for this classification problem, then what if we apply CNN?

Simple answer is too much for the provided dataset as is.
But it would be effective if:

- We need a single model that needs to work fine in different labs. Then the model needs to generalize well.
- And we _have enough data_ from variety of laboratories & its variations.

This example has three notebooks, follow below to try this.

1. Run followings.
    ```sh
    cd apps/cnn-laser-machine-listener
    ./download.sh
    ```
2. Preprocess data by CNN-LML-Preprocess-Data.ipynb.
3. Train model by CNN-LML-Train.ipynb.
4. Convert model to .pb by CNN-LML-TF-Model-Conversion.ipynb.
5. You can predict in realtime as follows.
    ```sh
    python predict_this.py
    ```

`cnn-model-laser-machine-listener.pb` is provided for quick try.
