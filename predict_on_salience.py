import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mir_eval

import os
import glob
import csv
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


'''Evaluation of three real recordings
'''

# # ## EVAL LOCUS ISTE FROM DCS
# sal_dir = './data/generalisation/LocusIste/LateDeep/DCS_LocusIste_mixture.npy'
# mf0_est_dir = './data/generalisation/LocusIste/LateDeep/DCS_LocusIste_mixture.csv'
# mf0_ref_dir = './data/generalisation/LocusIste/DCS_LocusIste_mixture.csv'
#
# salience = load_salience_function(sal_dir)
#
# ref_time, ref_s, ref_a, ref_t, ref_b, mf0 = load_mf0_reference(mf0_ref_dir)
#
# voascnn = models.voasCNN(PATCH_LEN)
# voascnn.load_weights("./models/cnn_full.h5")
# est_saliences = predict_one_example(salience, voascnn)
#
# thresh=0.5
#
# timestamp, sop = utils.pitch_activations_to_mf0(est_saliences[0], thresh=thresh)
# _, alt = utils.pitch_activations_to_mf0(est_saliences[1], thresh=thresh)
# _, ten = utils.pitch_activations_to_mf0(est_saliences[2], thresh=thresh)
# _, bas = utils.pitch_activations_to_mf0(est_saliences[3], thresh=thresh)
#
# # construct the multi-pitch predictions
# predictions = np.zeros([len(timestamp), 5])
#
#
# min_vals = [
#     np.min(sop[np.where(sop>0)[0]]), np.min(alt[np.where(alt>0)[0]]), np.min(ten[np.where(ten>0)[0]]), np.min(bas[np.where(bas>0)[0]])
# ]
# sorted_args = np.argsort(min_vals)
#
# predictions[:, 0] = timestamp
# predictions[:, 1] = sop
# predictions[:, 2] = alt
# predictions[:, 3] = ten
# predictions[:, 4] = bas
#
#
# # max_vals = np.median(predictions[:, 1:], axis=0)
# # sorted_args = np.argsort(max_vals)
#
#
# est_time = predictions[:, 0]
# # store already SATB
# # est_s_ = list(predictions[:, sorted_args[3]+1])
# # est_t_ = list(predictions[:, sorted_args[1]+1])
# # est_a_ = list(predictions[:, sorted_args[2]+1])
# # est_b_ = list(predictions[:, sorted_args[0]+1])
#
# est_s_ = list(predictions[:, 1])
# est_a_ = list(predictions[:, 2])
# est_t_ = list(predictions[:, 3])
# est_b_ = list(predictions[:, 4])
#
# est_b = []
# for row in est_b_: est_b.append(np.array([row]))
# est_t = []
# for row in est_t_: est_t.append(np.array([row]))
# est_a = []
# for row in est_a_: est_a.append(np.array([row]))
# est_s = []
# for row in est_s_: est_s.append(np.array([row]))
#
#
# # est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])
# #
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_s)):
#     if fqs == 0:
#         est_s[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_s = mir_eval.multipitch.evaluate(ref_time, ref_s, est_time, est_s)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_a)):
#     if fqs == 0:
#         est_a[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_a = mir_eval.multipitch.evaluate(ref_time, ref_a, est_time, est_a)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_t)):
#     if fqs == 0:
#         est_t[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_t = mir_eval.multipitch.evaluate(ref_time, ref_t, est_time, est_t)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_b)):
#     if fqs == 0:
#         est_b[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_b = mir_eval.multipitch.evaluate(ref_time, ref_b, est_time, est_b)
#
#
# print("Results VA for Locus Iste from DCS")
# print("----------------------------------")
# print("Soprano F-Score = {}".format(2*(metrics_s["Precision"]*metrics_s["Recall"])/(metrics_s["Precision"]+metrics_s["Recall"])))
# print("Alto F-Score = {}".format(2*(metrics_a["Precision"]*metrics_a["Recall"])/(metrics_a["Precision"]+metrics_a["Recall"])))
# print("Tenor F-Score = {}".format(2*(metrics_t["Precision"]*metrics_t["Recall"])/(metrics_t["Precision"]+metrics_t["Recall"])))
# print("Bass F-Score = {}".format(2*(metrics_b["Precision"]*metrics_b["Recall"])/(metrics_b["Precision"]+metrics_b["Recall"])))
# #
# #
# ## EVAL ROSSINYOL FROM CSD
# sal_dir = './data/generalisation/ElRossinyol/LateDeep/CSD_ER_mixture.npy'
# mf0_est_dir = './data/generalisation/ElRossinyol/LateDeep/CSD_ER_mixture.csv'
# mf0_ref_dir = './data/generalisation/ElRossinyol/CSD_ER_mixture.csv'
#
# salience = load_salience_function(sal_dir)
#
# ref_time, ref_s, ref_a, ref_t, ref_b, mf0 = load_mf0_reference(mf0_ref_dir)
#
# voascnn = models.voasCNN(PATCH_LEN)
# voascnn.load_weights("./models/cnn_full.h5")
# est_saliences = predict_one_example(salience, voascnn)
#
# thresh=0.4
#
# timestamp, sop = utils.pitch_activations_to_mf0(est_saliences[0], thresh=thresh)
# _, alt = utils.pitch_activations_to_mf0(est_saliences[1], thresh=thresh)
# _, ten = utils.pitch_activations_to_mf0(est_saliences[2], thresh=thresh)
# _, bas = utils.pitch_activations_to_mf0(est_saliences[3], thresh=thresh)
#
# # construct the multi-pitch predictions
# predictions = np.zeros([len(timestamp), 5])
#
#
# min_vals = [
#     np.min(sop[np.where(sop>0)[0]]), np.min(alt[np.where(alt>0)[0]]), np.min(ten[np.where(ten>0)[0]]), np.min(bas[np.where(bas>0)[0]])
# ]
# sorted_args = np.argsort(min_vals)
#
# predictions[:, 0] = timestamp
# predictions[:, 1] = sop
# predictions[:, 2] = alt
# predictions[:, 3] = ten
# predictions[:, 4] = bas
#
#
# # max_vals = np.median(predictions[:, 1:], axis=0)
# # sorted_args = np.argsort(max_vals)
# '''plotting
# '''
# plt.figure(figsize=(20,15))
# plt.subplot(221), plt.plot(mf0[:, 0], mf0[:, 3], '.k', markersize=10, label='GT'), \
# plt.title("$\hat{F0}_S$", fontsize=30)
# plt.xlim([4, 14]), plt.ylim([200, 800]), plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.xticks(fontsize=15), plt.yticks(fontsize=15)
# plt.subplot(222), plt.plot(mf0[:, 0], mf0[:, 1], '.k', markersize=10, label='GT'), plt.title("$\hat{F0}_A$", fontsize=30)
# plt.xlim([4, 14]), plt.ylim([150, 800])
# plt.xticks(fontsize=15), plt.yticks(fontsize=15)
# plt.subplot(223), plt.plot(mf0[:, 0], mf0[:, 4], '.k', markersize=10, label='GT'), plt.title("$\hat{F0}_T$", fontsize=30)
# plt.xlim([4, 14]), plt.ylim([100, 500])
# plt.xticks(fontsize=15), plt.yticks(fontsize=15), plt.xlabel("Time (sec)", fontsize=20)
# plt.subplot(224), plt.plot(mf0[:, 0], mf0[:, 2], '.k', markersize=10, label='GT'), plt.title("$\hat{F0}_B$", fontsize=30)
# plt.xlim([4, 14]), plt.xlabel("Time (sec)", fontsize=20), plt.ylim([50, 350]), plt.ylabel("Frequency (Hz)", fontsize=20)
# plt.xticks(fontsize=15), plt.yticks(fontsize=15)
#
# est_time = predictions[:, 0]
# # store already SATB
# # est_s_ = list(predictions[:, sorted_args[3]+1])
# # est_t_ = list(predictions[:, sorted_args[1]+1])
# # est_a_ = list(predictions[:, sorted_args[2]+1])
# # est_b_ = list(predictions[:, sorted_args[0]+1])
#
# est_s_ = list(predictions[:, 1])
# est_a_ = list(predictions[:, 2])
# est_t_ = list(predictions[:, 3])
# est_b_ = list(predictions[:, 4])
#
# plt.subplot(221), plt.plot(est_time, est_s_, '.', color='cadetblue', markersize=5, label='Prediction'), plt.legend(fontsize=20)
# plt.subplot(222), plt.plot(est_time, est_a_, '.', color='cadetblue', markersize=5, label='Prediction')
# plt.subplot(223), plt.plot(est_time, est_t_, '.', color='cadetblue', markersize=5, label='Prediction')
# plt.subplot(224), plt.plot(est_time, est_b_, '.', color='cadetblue', markersize=5, label='Prediction')
#
# plt.tight_layout()
# plt.savefig("output_example_rossinyol.png")
#
# est_b = []
# for row in est_b_: est_b.append(np.array([row]))
# est_t = []
# for row in est_t_: est_t.append(np.array([row]))
# est_a = []
# for row in est_a_: est_a.append(np.array([row]))
# est_s = []
# for row in est_s_: est_s.append(np.array([row]))
#
#
# # est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])
# #
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_s)):
#     if fqs == 0:
#         est_s[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_s = mir_eval.multipitch.evaluate(ref_time, ref_s, est_time, est_s)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_a)):
#     if fqs == 0:
#         est_a[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_a = mir_eval.multipitch.evaluate(ref_time, ref_a, est_time, est_a)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_t)):
#     if fqs == 0:
#         est_t[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_t = mir_eval.multipitch.evaluate(ref_time, ref_t, est_time, est_t)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_b)):
#     if fqs == 0:
#         est_b[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_b = mir_eval.multipitch.evaluate(ref_time, ref_b, est_time, est_b)
#
#
# print("Results VA for El Rossinyol from CSD")
# print("----------------------------------")
# print("Soprano F-Score = {}".format(2*(metrics_s["Precision"]*metrics_s["Recall"])/(metrics_s["Precision"]+metrics_s["Recall"])))
# print("Alto F-Score = {}".format(2*(metrics_a["Precision"]*metrics_a["Recall"])/(metrics_a["Precision"]+metrics_a["Recall"])))
# print("Tenor F-Score = {}".format(2*(metrics_t["Precision"]*metrics_t["Recall"])/(metrics_t["Precision"]+metrics_t["Recall"])))
# print("Bass F-Score = {}".format(2*(metrics_b["Precision"]*metrics_b["Recall"])/(metrics_b["Precision"]+metrics_b["Recall"])))
# #
# #
# # ## EVAL DERGREIS FROM CSD
# sal_dir = './data/generalisation/DerGreis/LateDeep/ECS_DG_mixture.npy'
# mf0_est_dir = './data/generalisation/DerGreis/LateDeep/ECS_DG_mixture.csv'
# mf0_ref_dir = './data/generalisation/DerGreis/ECS_DG_mixture.csv'
#
# salience = load_salience_function(sal_dir)
#
# ref_time, ref_s, ref_a, ref_t, ref_b, mf0 = load_mf0_reference(mf0_ref_dir)
#
# voascnn = models.voasCNN(PATCH_LEN)
# voascnn.load_weights("./models/cnn_full.h5")
# est_saliences = predict_one_example(salience, voascnn)
#
# thresh=0.5
#
# timestamp, sop = utils.pitch_activations_to_mf0(est_saliences[0], thresh=thresh)
# _, alt = utils.pitch_activations_to_mf0(est_saliences[1], thresh=thresh)
# _, ten = utils.pitch_activations_to_mf0(est_saliences[2], thresh=thresh)
# _, bas = utils.pitch_activations_to_mf0(est_saliences[3], thresh=thresh)
#
# # construct the multi-pitch predictions
# predictions = np.zeros([len(timestamp), 5])
#
#
# min_vals = [
#     np.min(sop[np.where(sop>0)[0]]), np.min(alt[np.where(alt>0)[0]]), np.min(ten[np.where(ten>0)[0]]), np.min(bas[np.where(bas>0)[0]])
# ]
# sorted_args = np.argsort(min_vals)
#
# predictions[:, 0] = timestamp
# predictions[:, 1] = sop
# predictions[:, 2] = alt
# predictions[:, 3] = ten
# predictions[:, 4] = bas
#
#
# # max_vals = np.median(predictions[:, 1:], axis=0)
# # sorted_args = np.argsort(max_vals)
#
#
# est_time = predictions[:, 0]
# # store already SATB
# # est_s_ = list(predictions[:, sorted_args[3]+1])
# # est_t_ = list(predictions[:, sorted_args[1]+1])
# # est_a_ = list(predictions[:, sorted_args[2]+1])
# # est_b_ = list(predictions[:, sorted_args[0]+1])
#
# est_s_ = list(predictions[:, 1])
# est_a_ = list(predictions[:, 2])
# est_t_ = list(predictions[:, 3])
# est_b_ = list(predictions[:, 4])
#
# est_b = []
# for row in est_b_: est_b.append(np.array([row]))
# est_t = []
# for row in est_t_: est_t.append(np.array([row]))
# est_a = []
# for row in est_a_: est_a.append(np.array([row]))
# est_s = []
# for row in est_s_: est_s.append(np.array([row]))
#
#
# # est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])
# #
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_s)):
#     if fqs == 0:
#         est_s[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_s = mir_eval.multipitch.evaluate(ref_time, ref_s, est_time, est_s)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_a)):
#     if fqs == 0:
#         est_a[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_a = mir_eval.multipitch.evaluate(ref_time, ref_a, est_time, est_a)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_t)):
#     if fqs == 0:
#         est_t[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_t = mir_eval.multipitch.evaluate(ref_time, ref_t, est_time, est_t)
#
# # get rid of zeros in prediction for input to mir_eval
# for i, (tms, fqs) in enumerate(zip(est_time, est_b)):
#     if fqs == 0:
#         est_b[i] = np.array([])
#
# ## evaluate with monophonic streams
# metrics_b = mir_eval.multipitch.evaluate(ref_time, ref_b, est_time, est_b)
#
#
# print("Results VA for Der Greis from ECD")
# print("----------------------------------")
# print("Soprano F-Score = {}".format(2*(metrics_s["Precision"]*metrics_s["Recall"])/(metrics_s["Precision"]+metrics_s["Recall"])))
# print("Alto F-Score = {}".format(2*(metrics_a["Precision"]*metrics_a["Recall"])/(metrics_a["Precision"]+metrics_a["Recall"])))
# print("Tenor F-Score = {}".format(2*(metrics_t["Precision"]*metrics_t["Recall"])/(metrics_t["Precision"]+metrics_t["Recall"])))
# print("Bass F-Score = {}".format(2*(metrics_b["Precision"]*metrics_b["Recall"])/(metrics_b["Precision"]+metrics_b["Recall"])))


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

    basename = os.path.basename(salience_file)
    salience = load_salience_function(salience_file)

    ref_time, ref_s, ref_a, ref_t, ref_b, mf0 = load_mf0_reference(os.path.join(mf0_ref_dir, basename.replace(".npy", ".csv")))

    mode = "time"

    voascnn = models.voasConvLSTM(PATCH_LEN)
    voascnn.load_weights("./models/conv_lstm_full_degraded.h5")

    if mode == "time":
        est_saliences = predict_one_example(salience, voascnn, mode)

    else:
        est_saliences = predict_one_example(salience, voascnn, mode)


    thresh=[0.1, 0.3, 0.4, 0.4]

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
    # store already SATB
    # est_s_ = list(predictions[:, sorted_args[3]+1])
    # est_t_ = list(predictions[:, sorted_args[1]+1])
    # est_a_ = list(predictions[:, sorted_args[2]+1])
    # est_b_ = list(predictions[:, sorted_args[0]+1])

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


    # est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])
    #
    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(est_time, est_s)):
        if fqs == 0:
            est_s[i] = np.array([])

    ## evaluate with monophonic streams
    metrics_s = mir_eval.multipitch.evaluate(ref_time, ref_s, est_time, est_s)

    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(est_time, est_a)):
        if fqs == 0:
            est_a[i] = np.array([])

    ## evaluate with monophonic streams
    metrics_a = mir_eval.multipitch.evaluate(ref_time, ref_a, est_time, est_a)

    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(est_time, est_t)):
        if fqs == 0:
            est_t[i] = np.array([])

    ## evaluate with monophonic streams
    metrics_t = mir_eval.multipitch.evaluate(ref_time, ref_t, est_time, est_t)

    # get rid of zeros in prediction for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(est_time, est_b)):
        if fqs == 0:
            est_b[i] = np.array([])

    ## evaluate with monophonic streams
    metrics_b = mir_eval.multipitch.evaluate(ref_time, ref_b, est_time, est_b)

    metrics_s["FScore"] = 2*(metrics_s["Precision"]*metrics_s["Recall"])/(metrics_s["Precision"]+metrics_s["Recall"])
    metrics_a["FScore"] = 2 * (metrics_a["Precision"] * metrics_a["Recall"]) / (metrics_a["Precision"] + metrics_a["Recall"])
    metrics_t["FScore"] = 2 * (metrics_t["Precision"] * metrics_t["Recall"]) / (metrics_t["Precision"] + metrics_t["Recall"])
    metrics_b["FScore"] = 2 * (metrics_b["Precision"] * metrics_b["Recall"]) / (metrics_b["Precision"] + metrics_b["Recall"])
    print(
        2 * (metrics_s["Precision"] * metrics_s["Recall"]) / (metrics_s["Precision"] + metrics_s["Recall"]),
        2 * (metrics_a["Precision"] * metrics_a["Recall"]) / (metrics_a["Precision"] + metrics_a["Recall"]),
        2 * (metrics_t["Precision"] * metrics_t["Recall"]) / (metrics_t["Precision"] + metrics_t["Recall"]),
        2 * (metrics_b["Precision"] * metrics_b["Recall"]) / (metrics_b["Precision"] + metrics_b["Recall"])
    )


    metrics_s["track"] = salience_file
    metrics_a["track"] = salience_file
    metrics_t["track"] = salience_file
    metrics_b["track"] = salience_file

    all_metrics_s.append(metrics_s)
    all_metrics_a.append(metrics_a)
    all_metrics_t.append(metrics_t)
    all_metrics_b.append(metrics_b)

pd.DataFrame(all_metrics_s).to_csv("bsq_voasclstm_degraded_soprano.csv", index=False)
pd.DataFrame(all_metrics_a).to_csv("bsq_voasclstm_degraded_alto.csv", index=False)
pd.DataFrame(all_metrics_t).to_csv("bsq_voasclstm_degraded_tenor.csv", index=False)
pd.DataFrame(all_metrics_b).to_csv("bsq_voasclstm_degraded_bass.csv", index=False)

