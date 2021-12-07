import numpy as np
import pandas as pd

import os
import argparse
import sys

from voas import utils as utils
from voas import config as config
from voas import models as models

PATCH_LEN = 128


def grab_input_slices(input_mat):
    '''Input mat will be [num_features x patch_len]
    '''

    slice_start_times = np.arange(start=0, stop=input_mat.shape[-1], step=PATCH_LEN)

    batches = []

    for i in slice_start_times[:-1]:
        chunk = input_mat[:, i:i + PATCH_LEN]
        batches.append(chunk)

    last_chunk = np.zeros([config.num_features, PATCH_LEN])
    last_chunk[:, :input_mat[:, slice_start_times[-1]:].shape[-1]] = input_mat[:, slice_start_times[-1]:]
    batches.append(last_chunk)

    return batches

def grab_input_slices_lstm(input_mat):
    '''Input mat will be [num_features x patch_len]
    '''

    slice_start_times = np.arange(start=0, stop=input_mat.shape[-1], step=PATCH_LEN)

    batches = []

    for i in slice_start_times[:-1]:
        chunk = input_mat[:, i:i + PATCH_LEN].transpose()
        batches.append(chunk)

    last_chunk = np.zeros([config.num_features, PATCH_LEN])
    last_chunk[:, :input_mat[:, slice_start_times[-1]:].shape[-1]] = input_mat[:, slice_start_times[-1]:]
    batches.append(last_chunk.transpose())

    return batches


def eval_generator(data_batches):
    for batch in data_batches:
        yield batch[np.newaxis, :, :, np.newaxis]


def predict_one_example(input_mat, model, mode):

    if mode == "time":
        batches = grab_input_slices_lstm(input_mat)
        pred = model.predict(x=eval_generator(batches), verbose=1)
        T_orig = input_mat.shape[-1]
        T_pred = np.hstack(pred[0]).shape[0]
        diff = T_pred - T_orig

        # import pdb; pdb.set_trace()

        return np.vstack(pred[0]).transpose()[:, :-diff], \
               np.vstack(pred[1]).transpose()[:, :-diff], \
               np.vstack(pred[2]).transpose()[:, :-diff], \
               np.vstack(pred[-1]).transpose()[:, :-diff], \
               pred

    else:
        batches = grab_input_slices(input_mat)

        pred = model.predict(x=eval_generator(batches), verbose=1)

        T_orig = input_mat.shape[-1]
        T_pred = np.hstack(pred[0]).shape[-1]
        diff = T_pred - T_orig

        return np.hstack(pred[0])[:, :-diff], np.hstack(pred[1])[:, :-diff], np.hstack(
            pred[2])[:, :-diff], np.hstack(pred[-1])[:, :-diff], pred


def load_salience_function(path_to_salience):
    # assume npy format for the salience from Late/Deep CNN
    salience = np.load(path_to_salience)

    return salience


def predict_one_file(model, salience, thresholds):

    est_saliences = predict_one_example(salience, model, mode="freq")

    timestamp, sop = utils.pitch_activations_to_mf0(est_saliences[0], thresh=thresholds[0])
    _, alt = utils.pitch_activations_to_mf0(est_saliences[1], thresh=thresholds[1])
    _, ten = utils.pitch_activations_to_mf0(est_saliences[2], thresh=thresholds[2])
    _, bas = utils.pitch_activations_to_mf0(est_saliences[3], thresh=thresholds[3])

    # construct the multi-pitch predictions
    predictions = np.zeros([len(timestamp), 5])

    predictions[:, 0] = timestamp
    predictions[:, 1] = sop
    predictions[:, 2] = alt
    predictions[:, 3] = ten
    predictions[:, 4] = bas

    return predictions


def main(args):

    if args.model == "voas_cnn":
        thresholds = [0.1, 0.3, 0.4, 0.4]
        model = models.voasCNN(PATCH_LEN)
        model.load_weights("./models/voas_cnn.h5")

    elif args.model == "voas_clstm":
        thresholds = [0.1, 0.3, 0.4, 0.5]
        model = models.voasConvLSTM(PATCH_LEN)
        model.load_weights("./models/voas_clstm.h5")

    else:
        sys.exit("Please provide a valid model. Expected `voas_cnn` or `voas_clstm`.")


    if args.saliencefolder != 0:

        salience_folder = args.saliencefolder

        for salience_file in os.listdir(salience_folder):
            if not salience_file.endswith("npy"): continue

            salience = load_salience_function(salience_file)
            predictions = predict_one_file(model, salience, thresholds)

            if args.outputpath != "0":
                output_folder = args.outputpath
                pd.DataFrame(predictions).to_csv(
                    os.path.join(output_folder, "{}".format(salience_file.replace("npy", "csv"))), header=False, index=False, index_label=False
                )

            else:
                pd.DataFrame(predictions).to_csv(
                    os.path.join(salience_folder, "{}".format(salience_file.replace("npy", "csv"))), header=False,
                                 index=False, index_label=False
                )

    else:
        salience_file = args.saliencefile
        basename = os.path.basename(salience_file)
        salience = load_salience_function(salience_file)
        predictions = predict_one_file(model, salience, thresholds)

        if args.outputpath != "0":
            output_folder = args.outputpath

            pd.DataFrame(predictions).to_csv(
                os.path.join(output_folder, "{}".format(basename.replace("npy", "csv"))), header=False, index=False,
                             index_label=False
            )
        else:
            pd.DataFrame(predictions).to_csv(
                os.path.join(salience_file.replace("npy", "csv")), header=False, index=False, index_label=False
            )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict F0 contours given an input polyphonic pitch salience function.")

    parser.add_argument("--model",
                        dest='model',
                        type=str,
                        help="Model to use for prediction: voas_clstm | voas_cnn")

    parser.add_argument("--saliencefile",
                        type=str,
                        default=0,
                        help="Path to the input salience file. It expects a npy fils.")

    parser.add_argument("--saliencefolder",
                        type=str,
                        default=0,
                        help="Path to the folder with salience files.")

    parser.add_argument("--outputpath",
                        type=str,
                        default="0",
                        help="Path to the folder to store the results. If nothing is provided, results will be stored in the same folder of the input(s).")
    main(parser.parse_args())