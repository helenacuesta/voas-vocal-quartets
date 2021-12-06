import numpy as np
import pandas as pd

import mir_eval

import os
import glob
import argparse

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

def load_mf0_reference(path_to_mf0):

    mf0 = pd.read_csv(path_to_mf0, header=None, index_col=False).values

    median_vals = np.median(mf0[:, 1:], axis=0)
    sorted_args = np.argsort(median_vals)

    timebase = mf0[:, 0]
    ref_b_ = list(mf0[:, sorted_args[0]+1])
    ref_t_ = list(mf0[:, sorted_args[1]+1])
    ref_a_ = list(mf0[:, sorted_args[2]+1])
    ref_s_ = list(mf0[:, sorted_args[3]+1])

    ref_b = []
    for row in ref_b_: ref_b.append(np.array([row]))

    ref_t = []
    for row in ref_t_: ref_t.append(np.array([row]))

    ref_a = []
    for row in ref_a_: ref_a.append(np.array([row]))

    ref_s = []
    for row in ref_s_: ref_s.append(np.array([row]))


    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(timebase, ref_b)):
        if fqs == 0:
            ref_b[i] = np.array([])

    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(timebase, ref_t)):
        if fqs==0:
            ref_t[i] = np.array([])

    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(timebase, ref_a)):
        if fqs == 0:
            ref_a[i] = np.array([])

    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(timebase, ref_s)):
        if fqs == 0:
            ref_s[i] = np.array([])

    # we return SATB
    return timebase, ref_s, ref_a, ref_t, ref_b, mf0



'''Evaluation on BSQ
'''
sal_dir = './data/BQ/'
mf0_ref_dir = './data/BQ/groundtruth'

salience_files = glob.glob(os.path.join(sal_dir, "*.npy"))
ref_files = glob.glob(os.path.join(mf0_ref_dir, "*.csv"))

all_metrics_s = []
all_metrics_a = []
all_metrics_t = []
all_metrics_b = []

for salience_file in salience_files:




    mode = "freq"

    voascnn = models.voasConvLSTM(PATCH_LEN)
    voascnn.load_weights("./models/voas_clstm.h5")

    if mode == "time":
        est_saliences = predict_one_example(salience, voascnn, mode)

    else:
        est_saliences = predict_one_example(salience, voascnn, mode)


    # thresh=[0.1, 0.3, 0.4, 0.5]
    thresh = [0.36, 0.37, 0.38, 0.41]

    timestamp, sop = utils.pitch_activations_to_mf0(est_saliences[0], thresh=thresh[0])
    _, alt = utils.pitch_activations_to_mf0(est_saliences[1], thresh=thresh[1])
    _, ten = utils.pitch_activations_to_mf0(est_saliences[2], thresh=thresh[2])
    _, bas = utils.pitch_activations_to_mf0(est_saliences[3], thresh=thresh[3])

    # construct the multi-pitch predictions
    predictions = np.zeros([len(timestamp), 5])


    min_vals = [
        np.min(sop[np.where(sop>0)[0]]), np.min(alt[np.where(alt>0)[0]]), np.min(ten[np.where(ten>0)[0]]), np.min(bas[np.where(bas>0)[0]])
    ]
    sorted_args = np.argsort(min_vals)

    predictions[:, 0] = timestamp
    predictions[:, 1] = sop
    predictions[:, 2] = alt
    predictions[:, 3] = ten
    predictions[:, 4] = bas


    # max_vals = np.median(predictions[:, 1:], axis=0)
    # sorted_args = np.argsort(max_vals)


    est_time = predictions[:, 0]

    est_s_ = list(predictions[:, 1])
    est_a_ = list(predictions[:, 2])
    est_t_ = list(predictions[:, 3])
    est_b_ = list(predictions[:, 4])

    est_b = []
    for row in est_b_: est_b.append(np.array([row]))
    est_t = []
    for row in est_t_: est_t.append(np.array([row]))
    est_a = []
    for row in est_a_: est_a.append(np.array([row]))
    est_s = []
    for row in est_s_: est_s.append(np.array([row]))


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
        print("Please provide a valid model: `voas_cnn` or `voas_clstm`")



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
                        #dest='data_splits',
                        type=str,
                        default=0,
                        help="Path to the input salience file. It expects a npy fils.")

    parser.add_argument("--saliencefolder",
                        type=str,
                        default=0,
                        help="Path to the folder with salience files.")

    parser.add_argument("--extension",
                        type=str,
                        default="wav",
                        help="Audio format extension. Defaults to wav.")


    parser.add_argument("--outputpath",
                        type=str,
                        default="0",
                        help="Path to the folder to store the results. If nothing is provided, results will be stored in the same folder of the input(s).")
    main(parser.parse_args())