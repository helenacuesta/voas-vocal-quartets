import sys
sys.path.append('./')

from voas import utils, models, config
from voas import training as training

import argparse

RANDOM_STATE = 22

def main(args):

    '''Pipeline for training LSTM for VoAs
    '''

    model_name = args.model_name

    data_splits_path = args.data_splits
    data_splits = utils.load_json_data(data_splits_path)

    patch_len = args.patch_len

    epochs = args.num_epochs
    batch_size = args.batch_size
    steps_per_epoch = args.steps_per_epoch
    val_steps = args.val_steps

    model_type = args.model

    if model_type in ["voas_clstm_reverse", "voas_clstm_reverse_1d", "voas_cnn", "voas_cnn_small"]:
        mode = "freq"
    else:
        mode = "time"

    # create data splits
    # data_splits = utils.create_data_split(config.songs, data_splits_path)
    # print("Data splits created to {}".format(data_splits_path))

    # if args.model == 'lstm':
    #     # create model
    #     model = models.lstm(config.max_phr_len(patch_len), lstm_size=args.lstm_units)
    #
    # elif args.model == 'conv_lstm':
    #     # create model
    #     model = models.conv_lstm(config.max_phr_len(patch_len), lstm_size=args.lstm_units)
    #
    # elif args.model == "conv_lstm_new":
    #     model = models.convLSTM(config.max_phr_len(patch_len))
    #
    # elif args.model == "resnet_conv":
    #     model = models.pretrained_resnet_conv(config.max_phr_len(patch_len))
    #
    # else:
    #     raise ValueError("Please specify a valid model: lstm | conv_lstm")



    # train
    print("    >> training starts now...")

    model, history = training.train(model_type, name=model_name, data_splits=data_splits, patch_len=patch_len, epochs=epochs, batch_size=batch_size, steps_epoch=steps_per_epoch, val_steps=val_steps, mode=mode)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LSTM with num_units units for Voice Assignment.")

    parser.add_argument("--model",
                        dest='model',
                        type=str,
                        help="Model to train: lstm | conv_lstm | conv_lstm_new | voas_cnn")

    parser.add_argument("--name",
                        dest='model_name',
                        type=str,
                        help="Name for the experiment. Also used to save the model weights.")

    parser.add_argument("--data-splits",
                        dest='data_splits',
                        type=str,
                        help="Path to the data splits file..")

    parser.add_argument("--patch-len",
                        dest="patch_len",
                        type=int,
                        help="Input patch length.")

    parser.add_argument("--lstm-units",
                        dest='lstm_units',
                        type=int,
                        default=32,
                        help="Number of LSTM units in each layer. Defaults to 32.")

    parser.add_argument("--epochs",
                        dest="num_epochs",
                        type=int,
                        help="Number of epochs for training.")

    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=16,
                        help="Batch size to use in the experiments (int). Defaults to 16.")

    parser.add_argument("--steps_epoch",
                        dest="steps_per_epoch",
                        type=int,
                        default=2048)

    parser.add_argument("--val_steps",
                        dest="val_steps",
                        type=int,
                        default=512)

    main(parser.parse_args())
