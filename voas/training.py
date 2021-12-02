from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import tensorflow as tf

import os

import voas.config as config
import voas.pescador_generators as pescgen
import voas.utils as utils
import voas.optimize_thresh as opt_thresh
import voas.evaluate as evaluate
import voas.models as models

import pickle
import datetime

#tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

def config_callbacks(exp_name):
    # define callbacks
    model_ckp = ModelCheckpoint(
        filepath=os.path.join(config.models_dir, 'ckp_{}'.format(exp_name)),
        verbose=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    earlystop = EarlyStopping(
        patience=20, verbose=1
    )
    log_dir = os.path.join(config.models_dir, 'tensorboard/logs/fit/{}_{}'.format(exp_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    # tboard = TensorBoard(
    #     log_dir=log_dir, write_graph=True, update_freq='epoch'
    # )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', patience=5, verbose=1
    )

    return [model_ckp, earlystop, reduce_lr] #tboard, reduce_lr]


def train(model_type, name, data_splits, patch_len, epochs, batch_size, steps_epoch, val_steps, mode):

    mirrored_strategy = tf.distribute.MirroredStrategy()

    print("[MSG] >>>>> Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

    print("[MSG] >>>>> Instantiating train and validation generators...")

    if mode not in ["time", "freq"]:
        raise ValueError("Wrong mode. Expected `freq` or `time`, got {}".format(mode))

    # #
    if mode == "freq":

        train_generator = tf.data.Dataset.from_generator(
            pescgen.full_generator_pescador,
            args=(data_splits['train'], "freq", patch_len, batch_size),
            output_types=(tf.dtypes.float32, (tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32)),
            output_shapes=(tf.TensorShape([None, config.num_features, patch_len, 1]),
                           (tf.TensorShape([None, config.num_features, patch_len]), tf.TensorShape([None, config.num_features, patch_len]),
                            tf.TensorShape([None, config.num_features, patch_len]),tf.TensorShape([None, config.num_features, patch_len])
                            )
                           )
        )


        val_generator = tf.data.Dataset.from_generator(
            pescgen.full_generator_pescador,
            args=(data_splits['validate'], "freq", patch_len, batch_size),
            output_types=(tf.dtypes.float32, (tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32)),
            output_shapes=(tf.TensorShape([None, config.num_features, patch_len, 1]),
                           (tf.TensorShape([None, config.num_features, patch_len]), tf.TensorShape([None, config.num_features, patch_len]),
                            tf.TensorShape([None, config.num_features, patch_len]), tf.TensorShape([None, config.num_features, patch_len])
                            )
                           )
        )

    elif mode == "time":


        train_generator = tf.data.Dataset.from_generator(
            pescgen.full_generator_pescador,
            args=(data_splits['train'], "time", patch_len, batch_size),
            output_types=(tf.dtypes.float32, (tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32)),
            output_shapes=(tf.TensorShape([None, patch_len, config.num_features, 1]),
                           (tf.TensorShape([None, patch_len, config.num_features]), tf.TensorShape([None, patch_len, config.num_features]),
                            tf.TensorShape([None, patch_len, config.num_features]),tf.TensorShape([None, patch_len, config.num_features])
                            )
                           )
        )


        val_generator = tf.data.Dataset.from_generator(
            pescgen.full_generator_pescador,
            args=(data_splits['validate'], "time", patch_len, batch_size),
            output_types=(tf.dtypes.float32, (tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32)),
            output_shapes=(tf.TensorShape([None, patch_len, config.num_features, 1]),
                           (tf.TensorShape([None, patch_len, config.num_features]), tf.TensorShape([None, patch_len, config.num_features]),
                            tf.TensorShape([None, patch_len, config.num_features]), tf.TensorShape([None, patch_len, config.num_features])
                            )
                           )
        )


    callbacks = config_callbacks(name)

    ## build model

    with mirrored_strategy.scope():

        if model_type == "voas_clstm_reverse":
            model = models.voasConvLSTMReverse(config.max_phr_len(patch_len))

        elif model_type == "voas_clstm_reverse_1d":
            model = models.voasConvLSTM1DReverse(config.max_phr_len(patch_len))

        elif model_type == "voas_cnn":
            model = models.voasCNN(config.max_phr_len(patch_len))

        elif model_type == "conv_lstm":
            model = models.voasConvLSTM(config.max_phr_len(patch_len))

        else:
            raise ValueError("Please specify a valid model: voas_clstm_reverse | voas_clstm_reverse | voas_cnn | conv_lstm")


        opt = tf.keras.optimizers.Adam(learning_rate=config.init_lr)

        model.compile(
            optimizer=opt,
            loss=utils.bkld,
            metrics=['mse', 'accuracy']
        )

        print(model.summary(line_length=200))

        # load partially trained model to resume training
        model.load_weights(os.path.join(config.models_dir, 'ckp_{}'.format(name)))

        history = model.fit(
            x=train_generator.repeat(),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_generator.repeat(),
            steps_per_epoch=steps_epoch,
            validation_steps=val_steps

        )

        testing_model = model

        # # skipping training and loading checkpoint of trained model
        # testing_model.load_weights(os.path.join(config.models_dir, 'ckp_{}'.format(name)))

        # testing_model.load_weights(
        #     os.path.join(
        #         config.models_dir, "{}.h5".format(name)
        #     )
        # )
        # testing_model.save(
        #     os.path.join(
        #         config.models_dir, "{}.h5".format(name)
        #     )
        # )
    testing_model.save(
        os.path.join(
            config.models_dir, "{}.h5".format(name)
        )
    )
    # history = 0

    with open(os.path.join(config.models_dir, name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


    ## optimize threshold
    print("Optimizing threshold on the validation set...")
    # optimal_thresh = opt_thresh.optimize_threshold_full(data_splits["validate"], patch_len, testing_model, name, mode)
    optimal_thresholds = opt_thresh.optimize_threshold_individual(data_splits["validate"], patch_len, testing_model, name, mode)
    # optimal_thresholds = [0.1, 0.3, 0.4, 0.4]

    ## evaluation on test set
    print("Evaluation on the test set...")
    evaluate.evaluate_full_individual(data_splits["test"], testing_model, patch_len, name, optimal_thresholds, mode)
    # evaluate.evaluate_full(data_splits["test"], testing_model, patch_len, name, optimal_thresholds, mode)


    return model, history
