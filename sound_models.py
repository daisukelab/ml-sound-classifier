import keras
from keras.layers import Dense, Conv2D, SeparableConv2D, Convolution2D, AveragePooling2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization, Flatten, Input
from keras.models import Model, Sequential
from keras.applications.mobilenetv2 import MobileNetV2

def model_cnn_alexnet(input_shape, num_classes, time_compress=[2, 1, 1], early_strides=(2,3)):
    model = Sequential()
 
    model.add(Conv2D(48, 11,  input_shape=input_shape, strides=early_strides, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 5, strides=early_strides, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(192, 3, strides=(1, time_compress[0]), activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=(1, time_compress[1]), activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=(1, time_compress[2]), activation='relu', padding='same', name='last_conv'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def model_mobilenetv2(input_shape, num_classes):
    base_model = MobileNetV2(weights=None, input_shape=input_shape, include_top=False,
                            alpha=0.35, depth_multiplier=0.5)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_model(conf, weights=None, show_detail=False):
    if conf.model == 'alexnet':
        print('Model: AlexNet based')
        model = model_cnn_alexnet(conf.dims, conf.num_classes,
                                  time_compress=[1, 1, 1], early_strides=(3,2))
    else:
        print('Model: MobileNetV2')
        model = model_mobilenetv2(input_shape=conf.dims, num_classes=conf.num_classes)
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=conf.learning_rate),
              metrics=['accuracy'])
    if weights is not None:
        print('Loading weights:', weights)
        model.load_weights(weights, by_name=True, skip_mismatch=True)
    if show_detail:
        model.summary()
    return model

def freeze_model_layers(model, trainable_after_this=''):
    trainable = False
    for layer in model.layers:
        if layer.name == trainable_after_this:
            trainable = True
        layer.trainable = trainable
