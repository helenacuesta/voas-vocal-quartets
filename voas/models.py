from keras.models import Sequential

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, Conv2D, BatchNormalization, MaxPool2D, TimeDistributed, ConvLSTM2D, Dropout, Permute, Reshape, Flatten
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50

import voas.config as config

def voasCNN(patch_len):
    # x_in = Input(batch_shape=(config.batch_size, config.num_features, config.max_phr_len(patch_len), 1, 1))
    x_in = Input(shape=(config.num_features, config.max_phr_len(patch_len), 1))

    print("In shape = {}".format(x_in.shape))

    x = BatchNormalization()(x_in)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1"
    )(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2"
    )(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(
        filters=16,
        kernel_size=(70, 3),
        padding="same",
        activation="relu",
        name="conv_harm_1"
    )(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    #
    x = Conv2D(
        filters=16,
        kernel_size=(70, 3),
        padding="same",
        activation="relu",
        name="conv_harm_2"
    )(x)

    ## start four branches now

    x = BatchNormalization()(x)

    ## branch 1
    x1a = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1a"
    )(x)

    x1a = BatchNormalization()(x1a)

    x1b = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1b"
    )(x1a)



    ## branch 2
    x2a = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2a"
    )(x)

    x2a = BatchNormalization()(x2a)
    x2b = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2b"
    )(x2a)

    ## branch 3

    x3a = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv3a"
    )(x)

    x3a = BatchNormalization()(x3a)
    x3b = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv3b"
    )(x3a)

    x4a = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv4a"
    )(x)

    x4a = BatchNormalization()(x4a)
    x4b = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv4b"
    )(x4a)


    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano', padding='same', activation='sigmoid')(x1b)
    y1 = tf.squeeze(y1, axis=-1)
    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto', padding='same', activation='sigmoid')(x2b)
    y2 = tf.squeeze(y2, axis=-1)
    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor', padding='same', activation='sigmoid')(x3b)
    y3 = tf.squeeze(y3, axis=-1)
    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass', padding='same', activation='sigmoid')(x4b)
    y4 = tf.squeeze(y4, axis=-1)

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out, name='voasCNN')

    return model

def voasConvLSTM(patch_len):

    x_in = Input(shape=(config.num_features, config.max_phr_len(patch_len), 1))

    print("In shape = {}".format(x_in.shape))

    x = BatchNormalization()(x_in)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1"
    )(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters=16,
        kernel_size=(70, 3),
        padding="same",
        activation="relu",
        name="conv_harm_1"
    )(x)

    x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    #
    x = Conv2D(
        filters=16,
        kernel_size=(70, 3),
        padding="same",
        activation="relu",
        name="conv_harm_2"
    )(x)

    ## permute and add dimension for convlstm
    x = Permute((2, 1, 3))(x)
    x = tf.expand_dims(x, axis=-2)

    ## start four branches now

    x = BatchNormalization()(x)

    # branch 1
    x1 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_1_1"
    )(x)

    x1 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_1_2"
    )(x1)

    x1 = tf.squeeze(x1, axis=-2)

    # branch 2
    x2 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_2_1"
    )(x)

    x2 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_2_2"
    )(x2)

    x2 = tf.squeeze(x2, axis=-2)

    # branch 3
    x3 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_3_1"
    )(x)

    x3 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_3_2"
    )(x3)

    x3 = tf.squeeze(x3, axis=-2)

    # branch 4
    x4 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_4_1"
    )(x)

    x4 = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_4_2"
    )(x4)

    x4 = tf.squeeze(x4, axis=-2)

    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano', padding='same', activation='sigmoid')(x1)
    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto', padding='same', activation='sigmoid')(x2)
    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor', padding='same', activation='sigmoid')(x3)
    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass', padding='same', activation='sigmoid')(x4)

    y1, y2, y3, y4 = tf.squeeze(y1, axis=-1), tf.squeeze(y2, axis=-1), tf.squeeze(y3, axis=-1), tf.squeeze(y4, axis=-1)

    y1 = Permute((2, 1))(y1)
    y2 = Permute((2, 1))(y2)
    y3 = Permute((2, 1))(y3)
    y4 = Permute((2, 1))(y4)

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out)

    return model



