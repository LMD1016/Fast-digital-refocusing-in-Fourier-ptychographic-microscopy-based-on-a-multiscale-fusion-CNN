from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.initializers import glorot_uniform
from keras import backend as keras


def DRN(pretrained_weights=None, input_shape=(128, 128, 9)):
    model = Sequential()

    # Encoder part
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    # Final layers
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.summary()

    # Load pretrained weights if provided
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def module(input_layer, conv_num):
    # General convolution layer
    x1 = Conv2D(conv_num, (3, 3), activation='relu', padding='same')(input_layer)
    # Dilated convolution layer
    x2 = Conv2D(conv_num, (3, 3), dilation_rate=3, activation='relu', padding='same')(x1)
    # Cascade layer
    x = concatenate([x1, x2], axis=3)
    x = Conv2D(conv_num, (3, 3), activation='relu', padding='same')(x)
    return x


def DRNv2(pretrained_weights=None, input_shape=(128, 128, 9)):
    input_layer = Input(shape=input_shape)

    # Encoder part using modules
    x = module(input_layer, 64)
    x = MaxPooling2D((2, 2))(x)
    x = module(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = module(x, 256)
    x = Conv2D(256, (3, 3), padding='same')(x)

    # Flatten and output layers
    x = Flatten()(x)
    output_layer = Dense(16, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    # Load pretrained weights if provided
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model











