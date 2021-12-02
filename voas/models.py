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


def conv_lstm_2d(patch_len):
    # input shape (samples, time, rows, cols, channels)

    # x_in = Input(shape=(config.max_phr_len(patch_len), config.num_features, 1, 1), batch_shape=(config.batch_size, config.max_phr_len(patch_len), config.num_features, 1, 1))
    # x_in = Input(batch_shape=(config.batch_size, config.max_phr_len(patch_len), config.num_features, 1, 1))
    x_in = Input(shape=(patch_len, 360, 1, 1))

    x = BatchNormalization()(x_in)

    x = ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm1"
    )(x)

    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        return_sequences=True,
        name="convlstm2"
    )(x)

    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        return_sequences=True,
        name="convlstm3"
    )(x)

    ## BRANCHES START HERE

    ## branch 1

    x1 = ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x1_1"
    )(x)

    x1 = ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x1_2"
    )(x1)

    ## branch 2

    x2 = ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x2_1"
    )(x)

    x2 = ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x2_2"
    )(x2)

    ## branch 3

    x3 = ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x3_1"
    )(x)

    x3 = ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x3_2"
    )(x3)

    ## branch 4

    x4 = ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x4_1"
    )(x)

    x4 = ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm_x4_2"
    )(x4)

    x1, x2, x3, x4 = tf.squeeze(x1, axis=-2), tf.squeeze(x2, axis=-2), tf.squeeze(x3, axis=-2), tf.squeeze(x4, axis=-2)

    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano', padding='same', activation='sigmoid')(x1)
    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto', padding='same', activation='sigmoid')(x2)
    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor', padding='same', activation='sigmoid')(x3)
    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass', padding='same', activation='sigmoid')(x4)

    y1, y2, y3, y4 = tf.squeeze(y1, axis=-1), tf.squeeze(y2, axis=-1), tf.squeeze(y3, axis=-1), tf.squeeze(y4, axis=-1)

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out)

    return model

def voasConvLSTM(patch_len):
    # input shape (samples, time, rows, cols, channels)

    # x_in = Input(shape=(config.max_phr_len(patch_len), config.num_features, 1, 1), batch_shape=(config.batch_size, config.max_phr_len(patch_len), config.num_features, 1, 1))
    # x_in = Input(batch_shape=(config.batch_size, config.max_phr_len(patch_len), config.num_features, 1, 1))
    x_in = Input(shape=(patch_len, 360, 1, 1))

    x = BatchNormalization()(x_in)

    x = ConvLSTM2D(
        filters=32,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        # stateful=True,
        return_sequences=True,
        name="convlstm1"
    )(x)


    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        return_sequences=True,
        name="convlstm2"
    )(x)

    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 7),
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="tanh",
        return_sequences=True,
        name="convlstm3"
    )(x)

    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = tf.squeeze(x, axis=-2)

    ## BRANCHES START HERE

    ## branch 1

    x1 = Conv2D(
        filters=64,
        padding="same",
        dilation_rate=(2, 1),
        kernel_size=(3, 10),
        activation='relu',
        name="conv1_1"
    )(x)

    x1 = BatchNormalization()(x1)

    x1 = Conv2D(
        filters=32,
        padding="same",
        kernel_size=(5, 5),
        activation='relu',
        name="conv1_2"
    )(x1)

    x1 = BatchNormalization()(x1)

    x1 = Conv2D(
        filters=16,
        padding="same",
        kernel_size=(3, 3),
        activation='relu',
        name="conv1_3"
    )(x1)

    x1 = BatchNormalization()(x1)

    ## branch 2

    x2 = Conv2D(
        filters=64,
        padding="same",
        dilation_rate=(2, 1),
        kernel_size=(3, 10),
        activation='relu',
        name="conv2_1"
    )(x)

    x2 = BatchNormalization()(x2)

    x2 = Conv2D(
        filters=32,
        padding="same",
        kernel_size=(5, 5),
        activation='relu',
        name="conv2_2"
    )(x2)

    x2 = BatchNormalization()(x2)

    x2 = Conv2D(
        filters=16,
        padding="same",
        kernel_size=(3, 3),
        activation='relu',
        name="conv2_3"
    )(x2)

    x2 = BatchNormalization()(x2)

    ## branch 3

    x3 = Conv2D(
        filters=64,
        padding="same",
        dilation_rate=(2, 1),
        kernel_size=(3, 10),
        activation='relu',
        name="conv3_1"
    )(x)

    x3 = BatchNormalization()(x3)

    x3 = Conv2D(
        filters=32,
        padding="same",
        kernel_size=(5, 5),
        activation='relu',
        name="conv3_2"
    )(x3)

    x3 = BatchNormalization()(x3)

    x3 = Conv2D(
        filters=16,
        padding="same",
        kernel_size=(3, 3),
        activation='relu',
        name="conv3_3"
    )(x3)

    x3 = BatchNormalization()(x3)

    ## branch 4

    x4 = Conv2D(
        filters=64,
        padding="same",
        dilation_rate=(2, 1),
        kernel_size=(3, 10),
        activation='relu',
        name="conv4_1"
    )(x)

    x4 = BatchNormalization()(x4)

    x4 = Conv2D(
        filters=32,
        padding="same",
        kernel_size=(5, 5),
        activation='relu',
        name="conv4_2"
    )(x4)

    x4 = BatchNormalization()(x4)

    x4 = Conv2D(
        filters=16,
        padding="same",
        kernel_size=(3, 3),
        activation='relu',
        name="conv4_3"
    )(x4)

    x4 = BatchNormalization()(x4)

    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano', padding='same', activation='sigmoid')(x1)
    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto', padding='same', activation='sigmoid')(x2)
    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor', padding='same', activation='sigmoid')(x3)
    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass', padding='same', activation='sigmoid')(x4)

    y1, y2, y3, y4 = tf.squeeze(y1, axis=-1), tf.squeeze(y2, axis=-1), tf.squeeze(y3, axis=-1), tf.squeeze(y4, axis=-1)

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out)

    return model


def voasCNN_small(patch_len):
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

    # x = Conv2D(
    #     filters=32,
    #     kernel_size=(3, 3),
    #     padding="same",
    #     activation="relu",
    #     name="conv2"
    # )(x)

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
    # x = Conv2D(
    #     filters=16,
    #     kernel_size=(70, 3),
    #     padding="same",
    #     activation="relu",
    #     name="conv_harm_2"
    # )(x)

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

    model = Model(inputs=x_in, outputs=out, name='voasCNN_small')

    return model


def voasConvLSTMReverse(patch_len):

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

def voasConvLSTM1DReverse(patch_len):

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

    ## start four branches now

    x = BatchNormalization()(x)

    # branch 1
    x1 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        # stateful=True,
        return_sequences=True,
        name="convlstm_1_1"
    )(x)

    x1 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        return_sequences=True,
        name="convlstm_1_2"
    )(x1)



    # branch 2
    x2 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        return_sequences=True,
        name="convlstm_2_1"
    )(x)

    x2 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        # stateful=True,
        return_sequences=True,
        name="convlstm_2_2"
    )(x2)

    # branch 3
    x3 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        return_sequences=True,
        name="convlstm_3_1"
    )(x)

    x3 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        # stateful=True,
        return_sequences=True,
        name="convlstm_3_2"
    )(x3)

    # branch 4
    x4 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        return_sequences=True,
        name="convlstm_4_1"
    )(x)

    x4 = ConvLSTM1D(
        filters=32,
        kernel_size=35,
        padding="same",
        recurrent_activation="hard_sigmoid",
        activation="relu",
        return_sequences=True,
        name="convlstm_4_2"
    )(x4)

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

