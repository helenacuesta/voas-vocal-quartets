import sys
sys.path.append('./')

from voas import utils, models, config
from voas import training as training

import argparse

RANDOM_STATE = 22

def main(args):

    '''Pipeline for training models for voice assignment
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

    if model_type in ["voas_clstm", "voas_cnn"]:
        mode = "freq"
    else:
        mode = "time"

    # create data splits
    # data_splits = utils.create_data_split(config.songs, data_splits_path)
    # print("Data splits created to {}".format(data_splits_path))



    # train
    print("    >> training starts now...")

    model, history = training.train(model_type, name=model_name, data_splits=data_splits, patch_len=patch_len, epochs=epochs, batch_size=batch_size, steps_epoch=steps_per_epoch, val_steps=val_steps, mode=mode)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VoasCNN or VoasCLSTM for voice assignment of vocal quartets.")

    parser.add_argument("--model",
                        dest='model',
                        type=str,
                        help="Model to train: voas_clstm | voas_cnn")

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
