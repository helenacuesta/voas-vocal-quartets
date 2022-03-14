from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import tensorflow as tf

import os

import voas.config as config
import voas.pescador_generators as pescgen
import voas.utils as utils
import voas.optimize_thresh as opt_thresh
import voas.evaluate as evaluate
import voas.models as models

import datetime


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


def train(model_type, name, data_splits, patch_len, epochs, batch_size, steps_epoch, val_steps):

    mirrored_strategy = tf.distribute.MirroredStrategy()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    print("[MSG] >>>>> Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

    print("[MSG] >>>>> Instantiating train and validation generators...")


    ## declare generators for train and validations sets
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

    callbacks = config_callbacks(name)

    ## build model with mirrored strategy for parallel gpu training

    with mirrored_strategy.scope():

        if model_type == "voas_clstm":
            model = models.voasConvLSTM(config.max_phr_len(patch_len))

        elif model_type in ["voas_cnn", "voas_cnn_clean"]:
            model = models.voasCNN(config.max_phr_len(patch_len))

        else:
            raise ValueError("Please specify a valid model: voas_clstm| voas_cnn")


        opt = tf.keras.optimizers.Adam(learning_rate=config.init_lr)

        model.compile(
            optimizer=opt,
            loss=utils.bkld,
            metrics=['mse', 'accuracy']
        )

        print(model.summary(line_length=200))

        history = model.fit(
            x=train_generator.repeat(),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_generator.repeat(),
            steps_per_epoch=steps_epoch,
            validation_steps=val_steps

        )

        # # skip training, load checkpoint, and save
        # model.load_weights(os.path.join(config.models_dir, 'ckp_{}'.format(name)))
        #
        # model.save(
        #     os.path.join(
        #         config.models_dir, "{}.h5".format(name)
        #     )
        # )

        # # skip training and load trained model
        # model.load_weights(
        #     os.path.join(
        #         config.models_dir, "{}.h5".format(model_type)
        #     )
        # )

    # model.save(
    #     os.path.join(
    #         config.models_dir, "{}.h5".format(name)
    #     )
    # )
    # history = 0

    # with open(os.path.join(config.models_dir, name), 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)


    ## optimize threshold
    print("Optimizing threshold on the validation set...")

    optimal_thresholds = opt_thresh.optimize_threshold_individual(data_splits["validate"], patch_len, model, name)


    ## evaluation on test set
    print("Evaluation on the test set...")
    evaluate.evaluate_full_individual(data_splits["test"], model, patch_len, name, optimal_thresholds)


    return model, history
