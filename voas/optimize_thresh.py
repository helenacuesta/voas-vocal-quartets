import numpy as np
import pandas as pd

import mir_eval

import os
import glob
import csv

from voas import utils as utils
from voas import config as config


def optimize_threshold_full(validation_files, patch_len, model, exp_name, mode):
    '''Optimize detection threshold on the validation set according to multi-pitch metrics.
    We select one single threshold for all voices
    '''
    for song in validation_files:

        try:

            print("Processing file {}".format(song))

            ## load input/outputs for the song

            mat_mix = []
            with open(os.path.join(
                    config.feats_dir, "{}_mix.csv".format(song))
                    , "r") as f:
                rd = csv.reader(f)
                for line in rd: mat_mix.append(np.float32(line))
            mat_mix = np.array(mat_mix)


            mat_S = []
            with open(os.path.join(
                    config.feats_dir, "{}_S.csv".format(song))
                    , "r") as f:
                rd = csv.reader(f)
                for line in rd: mat_S.append(np.float32(line))
            mat_S = np.array(mat_S)

            mat_A = []
            with open(os.path.join(
                    config.feats_dir, "{}_A.csv".format(song))
                    , "r") as f:
                rd = csv.reader(f)
                for line in rd: mat_A.append(np.float32(line))
            mat_A = np.array(mat_A)

            mat_T = []
            with open(os.path.join(
                    config.feats_dir, "{}_T.csv".format(song))
                    , "r") as f:
                rd = csv.reader(f)
                for line in rd: mat_T.append(np.float32(line))
            mat_T = np.array(mat_T)

            mat_B = []
            with open(os.path.join(
                    config.feats_dir, "{}_B.csv".format(song))
                    , "r") as f:
                rd = csv.reader(f)
                for line in rd: mat_B.append(np.float32(line))
            mat_B = np.array(mat_B)

            print("Input/output files loaded correctly.")

            time_len = mat_mix.shape[1]

            if time_len <= patch_len:
                pass

            start_idx = np.arange(0, time_len - patch_len, step=patch_len)

            if mode == "freq":

                # input_mat = mat_mix
                rearr_input = []

                for idx in start_idx:
                    rearr_input.append(mat_mix[:, idx:idx+patch_len, np.newaxis])
                rearr_input = np.array(rearr_input)

                del mat_mix

                prediction_mat = model.predict(rearr_input)

                pred_s = np.hstack(prediction_mat[0])
                pred_a = np.hstack(prediction_mat[1])
                pred_t = np.hstack(prediction_mat[2])
                pred_b = np.hstack(prediction_mat[3])

                num_frames_pred = pred_s.shape[1]

                # we're not dealing with the last frame, so we cut the reference here
                mat_S = mat_S[:, :num_frames_pred]
                mat_A = mat_A[:, :num_frames_pred]
                mat_T = mat_T[:, :num_frames_pred]
                mat_B = mat_B[:, :num_frames_pred]

                ## convert reference mats to pitches
                ref_times_s, ref_freqs_s = utils.pitch_activations_to_mf0(mat_S, thresh=0.9)

                ref_times_a, ref_freqs_a = utils.pitch_activations_to_mf0(mat_A, thresh=0.9)

                ref_times_t, ref_freqs_t = utils.pitch_activations_to_mf0(mat_T, thresh=0.9)

                ref_times_b, ref_freqs_b = utils.pitch_activations_to_mf0(mat_B, thresh=0.9)

                del mat_S, mat_A, mat_T, mat_B

                # construct the multi-pitch reference

                reference = np.zeros([len(ref_times_s), 5])
                reference[:, 0] = ref_times_s
                reference[:, 1] = ref_freqs_s
                reference[:, 2] = ref_freqs_a
                reference[:, 3] = ref_freqs_t
                reference[:, 4] = ref_freqs_b

                ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])

                # get rid of zeros in prediction for input to mir_eval
                for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
                    if any(fqs == 0):
                        ref_freqs[i] = np.array([f for f in fqs if f > 0])

                thresh_vals = np.arange(0.1, 1.0, 0.1)
                thresh_scores = {t: [] for t in thresh_vals}

                print("Threshold optimization starts now...")

                for thresh in thresh_vals:

                    # print("Testing with thresh={}".format(thresh))

                    timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh)
                    _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh)
                    _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh)
                    _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh)

                    # construct the multi-pitch predictions
                    predictions = np.zeros([len(timestamp), 5])
                    predictions[:, 0] = timestamp
                    predictions[:, 1] = sop
                    predictions[:, 2] = alt
                    predictions[:, 3] = ten
                    predictions[:, 4] = bas

                    est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])

                    # get rid of zeros in prediction for input to mir_eval
                    for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                        if any(fqs == 0):
                            est_freqs[i] = np.array([f for f in fqs if f > 0])

                    metrics = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, max_freq=9000.0)
                    thresh_scores[thresh].append(metrics['Accuracy'])
                    # print("Moving to the next thresh...")


            elif mode == "time":

                mat_mix = mat_mix.transpose()
                rearr_input = []

                for idx in start_idx:
                    rearr_input.append(mat_mix[idx:idx+patch_len, :, np.newaxis])
                rearr_input = np.array(rearr_input)

                del mat_mix

                prediction_mat = model.predict(rearr_input)

                pred_s = np.vstack(prediction_mat[0]).transpose()
                pred_a = np.vstack(prediction_mat[1]).transpose()
                pred_t = np.vstack(prediction_mat[2]).transpose()
                pred_b = np.vstack(prediction_mat[3]).transpose()

                num_frames_pred = pred_s.shape[1]

                # we're not dealing with the last frame, so we cut the reference here
                mat_S = mat_S[:, :num_frames_pred]
                mat_A = mat_A[:, :num_frames_pred]
                mat_T = mat_T[:, :num_frames_pred]
                mat_B = mat_B[:, :num_frames_pred]

                ## convert reference mats to pitches
                ref_times_s, ref_freqs_s = utils.pitch_activations_to_mf0(mat_S, thresh=0.9)

                ref_times_a, ref_freqs_a = utils.pitch_activations_to_mf0(mat_A, thresh=0.9)

                ref_times_t, ref_freqs_t = utils.pitch_activations_to_mf0(mat_T, thresh=0.9)

                ref_times_b, ref_freqs_b = utils.pitch_activations_to_mf0(mat_B, thresh=0.9)

                del mat_S, mat_A, mat_T, mat_B

                # construct the multi-pitch reference

                reference = np.zeros([len(ref_times_s), 5])
                reference[:, 0] = ref_times_s
                reference[:, 1] = ref_freqs_s
                reference[:, 2] = ref_freqs_a
                reference[:, 3] = ref_freqs_t
                reference[:, 4] = ref_freqs_b

                ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])

                # get rid of zeros in prediction for input to mir_eval
                for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
                    if any(fqs == 0):
                        ref_freqs[i] = np.array([f for f in fqs if f > 0])

                thresh_vals = np.arange(0.1, 1.0, 0.1)
                thresh_scores = {t: [] for t in thresh_vals}

                print("Threshold optimization starts now...")

                for thresh in thresh_vals:

                    # print("Testing with thresh={}".format(thresh))

                    timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh)
                    _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh)
                    _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh)
                    _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh)

                    # construct the multi-pitch predictions
                    predictions = np.zeros([len(timestamp), 5])
                    predictions[:, 0] = timestamp
                    predictions[:, 1] = sop
                    predictions[:, 2] = alt
                    predictions[:, 3] = ten
                    predictions[:, 4] = bas

                    est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])

                    # get rid of zeros in prediction for input to mir_eval
                    for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                        if any(fqs == 0):
                            est_freqs[i] = np.array([f for f in fqs if f > 0])

                    metrics = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, max_freq=9000.0)
                    thresh_scores[thresh].append(metrics['Accuracy'])
                    # print("Moving to the next thresh...")

            else:
                raise ValueError("Wrong mode. Expected `time` or `freq` and got {}".format(mode))
        except:
            continue

    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print("Best threshold is {}".format(best_thresh))

    try:
        with open("{}_threshold.csv".format(exp_name), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Best threshold for joint optimization of all voices is: {}".format(best_thresh)])

    except:
        print("Best threshold is {}".format(best_thresh))

    return best_thresh


def optimize_threshold_individual(validation_files, patch_len, model, exp_name, mode):
    '''Optimize detection threshold on the validation set for each voice individually.
    The chosen thresholds maximize the overall accuracy for each voice.
    '''
    for song in validation_files:

        print("Processing file {}".format(song))

        ## load input/outputs for the song

        mat_mix = []
        with open(os.path.join(
                config.feats_dir, "{}_mix.csv".format(song))
                , "r") as f:
            rd = csv.reader(f)
            for line in rd: mat_mix.append(np.float32(line))
        mat_mix = np.array(mat_mix)


        mat_S = []
        with open(os.path.join(
                config.feats_dir, "{}_S.csv".format(song))
                , "r") as f:
            rd = csv.reader(f)
            for line in rd: mat_S.append(np.float32(line))
        mat_S = np.array(mat_S)

        mat_A = []
        with open(os.path.join(
                config.feats_dir, "{}_A.csv".format(song))
                , "r") as f:
            rd = csv.reader(f)
            for line in rd: mat_A.append(np.float32(line))
        mat_A = np.array(mat_A)

        mat_T = []
        with open(os.path.join(
                config.feats_dir, "{}_T.csv".format(song))
                , "r") as f:
            rd = csv.reader(f)
            for line in rd: mat_T.append(np.float32(line))
        mat_T = np.array(mat_T)

        mat_B = []
        with open(os.path.join(
                config.feats_dir, "{}_B.csv".format(song))
                , "r") as f:
            rd = csv.reader(f)
            for line in rd: mat_B.append(np.float32(line))
        mat_B = np.array(mat_B)

        print("Input/output files loaded correctly.")

        time_len = mat_mix.shape[1]

        if time_len <= patch_len:
            pass

        start_idx = np.arange(0, time_len - patch_len, step=patch_len)

        if mode == "freq":

            # input_mat = mat_mix
            rearr_input = []
            pred_s, pred_a, pred_t, pred_b = [], [], [], []

            for idx in start_idx:
                rearr_input.append(mat_mix[:, idx:idx+patch_len, np.newaxis])
            rearr_input = np.array(rearr_input)

            del mat_mix

            prediction_mat = model.predict(rearr_input)

            pred_s = np.hstack(prediction_mat[0])
            pred_a = np.hstack(prediction_mat[1])
            pred_t = np.hstack(prediction_mat[2])
            pred_b = np.hstack(prediction_mat[3])

            num_frames_pred = pred_s.shape[1]

            # we're not dealing with the last frame, so we cut the reference here
            mat_S = mat_S[:, :num_frames_pred]
            mat_A = mat_A[:, :num_frames_pred]
            mat_T = mat_T[:, :num_frames_pred]
            mat_B = mat_B[:, :num_frames_pred]

            ## convert reference mats to pitches
            ref_times_s, ref_freqs_s = utils.pitch_activations_to_mf0(mat_S, thresh=0.9)

            ref_times_a, ref_freqs_a = utils.pitch_activations_to_mf0(mat_A, thresh=0.9)

            ref_times_t, ref_freqs_t = utils.pitch_activations_to_mf0(mat_T, thresh=0.9)

            ref_times_b, ref_freqs_b = utils.pitch_activations_to_mf0(mat_B, thresh=0.9)

            del mat_S, mat_A, mat_T, mat_B

            # construct the multi-pitch reference

            # reference = np.zeros([len(ref_times_s), 5])
            # reference[:, 0] = ref_times_s
            # reference[:, 1] = ref_freqs_s
            # reference[:, 2] = ref_freqs_a
            # reference[:, 3] = ref_freqs_t
            # reference[:, 4] = ref_freqs_b
            #
            # ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])
            #
            # # get rid of zeros in prediction for input to mir_eval
            # for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            #     if any(fqs == 0):
            #         ref_freqs[i] = np.array([f for f in fqs if f > 0])

            thresh_vals = np.arange(0.1, 1.0, 0.1)
            thresh_scores_sop = {t: [] for t in thresh_vals}
            thresh_scores_alt = {t: [] for t in thresh_vals}
            thresh_scores_ten = {t: [] for t in thresh_vals}
            thresh_scores_bass = {t: [] for t in thresh_vals}

            print("Threshold optimization starts now...")

            for thresh in thresh_vals:

                # print("Testing with thresh={}".format(thresh))

                timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh)
                _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh)
                _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh)
                _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh)

                metrics_sop = mir_eval.melody.evaluate(ref_times_s, ref_freqs_s, timestamp, sop)
                metrics_alt = mir_eval.melody.evaluate(ref_times_a, ref_freqs_a, timestamp, alt)
                metrics_ten = mir_eval.melody.evaluate(ref_times_t, ref_freqs_t, timestamp, ten)
                metrics_bass = mir_eval.melody.evaluate(ref_times_b, ref_freqs_b, timestamp, bas)

                thresh_scores_sop[thresh].append(metrics_sop['Overall Accuracy'])
                thresh_scores_alt[thresh].append(metrics_alt['Overall Accuracy'])
                thresh_scores_ten[thresh].append(metrics_ten['Overall Accuracy'])
                thresh_scores_bass[thresh].append(metrics_bass['Overall Accuracy'])

            avg_thresh_sop = [np.mean(thresh_scores_sop[t]) for t in thresh_vals]
            avg_thresh_alt = [np.mean(thresh_scores_alt[t]) for t in thresh_vals]
            avg_thresh_ten = [np.mean(thresh_scores_ten[t]) for t in thresh_vals]
            avg_thresh_bass = [np.mean(thresh_scores_bass[t]) for t in thresh_vals]

            best_thresh_sop = thresh_vals[np.argmax(avg_thresh_sop)]
            best_thresh_alt = thresh_vals[np.argmax(avg_thresh_alt)]
            best_thresh_ten = thresh_vals[np.argmax(avg_thresh_ten)]
            best_thresh_bass = thresh_vals[np.argmax(avg_thresh_bass)]

            with open("{}_threshold_oa.csv".format(exp_name), "w") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Best thresholds for separate optimizations of all voices are: S={}, A={}, T={}, B={}".format(
                        best_thresh_sop,
                        best_thresh_alt,
                        best_thresh_ten,
                        best_thresh_bass
                    )])

            print("Best thresholds are {}".format(
                [best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass]))

            return best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass

            # except:
            #     print("Best thresholds are {}".format(
            #         [best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass]))
            #
            #     return best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass
            #
            # return best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass


        elif mode == "time":

            mat_mix = mat_mix.transpose()
            rearr_input = []

            for idx in start_idx:
                rearr_input.append(mat_mix[idx:idx+patch_len, :, np.newaxis])
            rearr_input = np.array(rearr_input)

            del mat_mix

            prediction_mat = model.predict(rearr_input)

            pred_s = np.vstack(prediction_mat[0]).transpose()
            pred_a = np.vstack(prediction_mat[1]).transpose()
            pred_t = np.vstack(prediction_mat[2]).transpose()
            pred_b = np.vstack(prediction_mat[3]).transpose()

            num_frames_pred = pred_s.shape[1]

            # we're not dealing with the last frame, so we cut the reference here
            mat_S = mat_S[:, :num_frames_pred]
            mat_A = mat_A[:, :num_frames_pred]
            mat_T = mat_T[:, :num_frames_pred]
            mat_B = mat_B[:, :num_frames_pred]

            ## convert reference mats to pitches
            ref_times_s, ref_freqs_s = utils.pitch_activations_to_mf0(mat_S, thresh=0.9)

            ref_times_a, ref_freqs_a = utils.pitch_activations_to_mf0(mat_A, thresh=0.9)

            ref_times_t, ref_freqs_t = utils.pitch_activations_to_mf0(mat_T, thresh=0.9)

            ref_times_b, ref_freqs_b = utils.pitch_activations_to_mf0(mat_B, thresh=0.9)

            del mat_S, mat_A, mat_T, mat_B

            # construct the multi-pitch reference

            # reference = np.zeros([len(ref_times_s), 5])
            # reference[:, 0] = ref_times_s
            # reference[:, 1] = ref_freqs_s
            # reference[:, 2] = ref_freqs_a
            # reference[:, 3] = ref_freqs_t
            # reference[:, 4] = ref_freqs_b
            #
            # ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])
            #
            # # get rid of zeros in prediction for input to mir_eval
            # for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            #     if any(fqs == 0):
            #         ref_freqs[i] = np.array([f for f in fqs if f > 0])

            thresh_vals = np.arange(0.1, 1.0, 0.1)
            thresh_scores_sop = {t: [] for t in thresh_vals}
            thresh_scores_alt = {t: [] for t in thresh_vals}
            thresh_scores_ten = {t: [] for t in thresh_vals}
            thresh_scores_bass = {t: [] for t in thresh_vals}

            print("Threshold optimization starts now...")

            for thresh in thresh_vals:

                # print("Testing with thresh={}".format(thresh))

                timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh)
                _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh)
                _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh)
                _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh)

                metrics_sop = mir_eval.melody.evaluate(ref_times_s, ref_freqs_s, timestamp, sop)
                metrics_alt = mir_eval.melody.evaluate(ref_times_a, ref_freqs_a, timestamp, alt)
                metrics_ten = mir_eval.melody.evaluate(ref_times_t, ref_freqs_t, timestamp, ten)
                metrics_bass = mir_eval.melody.evaluate(ref_times_b, ref_freqs_b, timestamp, bas)

                thresh_scores_sop[thresh].append(metrics_sop['Overall Accuracy'])
                thresh_scores_alt[thresh].append(metrics_alt['Overall Accuracy'])
                thresh_scores_ten[thresh].append(metrics_ten['Overall Accuracy'])
                thresh_scores_bass[thresh].append(metrics_bass['Overall Accuracy'])

            avg_thresh_sop = [np.mean(thresh_scores_sop[t]) for t in thresh_vals]
            avg_thresh_alt = [np.mean(thresh_scores_alt[t]) for t in thresh_vals]
            avg_thresh_ten = [np.mean(thresh_scores_ten[t]) for t in thresh_vals]
            avg_thresh_bass = [np.mean(thresh_scores_bass[t]) for t in thresh_vals]

            best_thresh_sop = thresh_vals[np.argmax(avg_thresh_sop)]
            best_thresh_alt = thresh_vals[np.argmax(avg_thresh_alt)]
            best_thresh_ten = thresh_vals[np.argmax(avg_thresh_ten)]
            best_thresh_bass = thresh_vals[np.argmax(avg_thresh_bass)]

            with open("{}_threshold_oa.csv".format(exp_name), "w") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Best thresholds for separate optimizations of all voices are: S={}, A={}, T={}, B={}".format(
                        best_thresh_sop,
                        best_thresh_alt,
                        best_thresh_ten,
                        best_thresh_bass
                    )])
            print("Best thresholds are {}".format(
                [best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass]))

            return best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass

            # except:
            #     print("Best thresholds are {}".format(
            #         [best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass]))
        else:
            raise ValueError("Wrong mode!")










