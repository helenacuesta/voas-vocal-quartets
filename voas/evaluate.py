import numpy as np
import pandas as pd

import mir_eval

import os
import glob
import csv

from voas import utils as utils
from voas import config as config


def evaluate_chunks(test_files, model, exp_name, thresh, mode):
    '''Evaluate the model on the test set, chunk-wise-
    '''

    all_metrics = []
    fscores = []
    for song in test_files:

        ## parse chunks
        list_of_chunks = sorted(
            glob.glob(os.path.join(config.feats_dir, "segmented", "{}_mix_*.csv".format(song)))
        )
        chunk_idx = np.arange(len(list_of_chunks))

        for idx in chunk_idx:


            ## load targets and convert to pitch

            ref_times_s , ref_freqs_s = utils.pitch_activations_to_mf0(
                pd.read_csv(
                    os.path.join(config.feats_dir, "segmented", "{}_S_{}.csv".format(song, idx),
                                 ), header=None, index_col=False
                ).values.transpose(), thresh=0.9
            )

            ref_times_a, ref_freqs_a = utils.pitch_activations_to_mf0(
                pd.read_csv(
                    os.path.join(config.feats_dir, "segmented", "{}_A_{}.csv".format(song, idx),
                                 ), header=None, index_col = False
                ).values.transpose(), thresh=0.9
            )

            ref_times_t, ref_freqs_t = utils.pitch_activations_to_mf0(
                pd.read_csv(
                    os.path.join(config.feats_dir, "segmented", "{}_T_{}.csv".format(song, idx)
                                 ), header=None, index_col=False
                ).values.transpose(), thresh=0.9
            )

            ref_times_b, ref_freqs_b = utils.pitch_activations_to_mf0(
                pd.read_csv(
                    os.path.join(config.feats_dir, "segmented", "{}_B_{}.csv".format(song, idx),
                                 ), header=None, index_col=False
                ).values.transpose(), thresh=0.9
            )

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

                    # load input representation

                    if mode == "time":

                        input_mat = pd.read_csv(
                            os.path.join(config.feats_dir, "segmented", "{}_mix_{}.csv".format(song, idx)
                                         ), header=None, index_col=False
                        ).values

                        # print("Shape of input mat for validation: {}".format(input_mat.shape))

                        prediction_mat = utils.get_single_chunk_prediction(model,
                                                                           input_mat[np.newaxis, :, :, np.newaxis,
                                                                           np.newaxis])

                    elif mode == "freq":

                        input_mat = pd.read_csv(
                            os.path.join(config.feats_dir, "segmented", "{}_mix_{}.csv".format(song, idx)
                                         ), header=None, index_col=False
                        ).values.transpose()

                        # print("Shape of input mat for validation: {}".format(input_mat.shape))

                        prediction_mat = utils.get_single_chunk_prediction(model,
                                                                           input_mat[np.newaxis, :, :, np.newaxis])

            if mode == "time":

                timestamp, sop = utils.pitch_activations_to_mf0(prediction_mat['sop'][0, :, :, 0].transpose(),
                                                                thresh=thresh)
                _, alt = utils.pitch_activations_to_mf0(prediction_mat['alt'][0, :, :, 0].transpose(), thresh=thresh)
                _, ten = utils.pitch_activations_to_mf0(prediction_mat['ten'][0, :, :, 0].transpose(), thresh=thresh)
                _, bas = utils.pitch_activations_to_mf0(prediction_mat['bas'][0, :, :, 0].transpose(), thresh=thresh)

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
                if metrics["Recall"] == 0.0:
                    Fscore = 0

                elif metrics["Precision"] == 0:
                    Fscore = 0

                else:
                    Fscore = 2 * (metrics["Recall"]*metrics["Precision"])/(metrics["Recall"]+metrics["Precision"])

                metrics["F-Score"] = Fscore
                fscores.append(Fscore)

                chunk_name = "{}_S_{}.csv".format(song, idx)
                metrics["track"] = chunk_name

                all_metrics.append(metrics)

            elif mode == "freq":
                timestamp, sop = utils.pitch_activations_to_mf0(prediction_mat['sop'][0, :, :, 0],
                                                                thresh=thresh)
                _, alt = utils.pitch_activations_to_mf0(prediction_mat['alt'][0, :, :, 0], thresh=thresh)
                _, ten = utils.pitch_activations_to_mf0(prediction_mat['ten'][0, :, :, 0], thresh=thresh)
                _, bas = utils.pitch_activations_to_mf0(prediction_mat['bas'][0, :, :, 0], thresh=thresh)

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
                if metrics["Recall"] == 0.0:
                    Fscore = 0

                elif metrics["Precision"] == 0:
                    Fscore = 0

                else:
                    Fscore = 2 * (metrics["Recall"] * metrics["Precision"]) / (metrics["Recall"] + metrics["Precision"])

                metrics["F-Score"] = Fscore
                fscores.append(Fscore)

                chunk_name = "{}_S_{}.csv".format(song, idx)
                metrics["track"] = chunk_name

                all_metrics.append(metrics)

            else:
                raise ValueError("Incorrect mode")

    pd.DataFrame(all_metrics).to_csv("{}_{}_all_scores.csv".format(exp_name, 'testset'))

    print("Average F-Score in the test set is: {}".format(np.mean(fscores)))

def evaluate_full(test_files, model, patch_len, exp_name, thresh, mode):
    '''Evaluate the model on the test set, full files
    '''

    all_metrics = []
    fscores = []

    ###
    for song in test_files:

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
                pred_s, pred_a, pred_t, pred_b = [], [], [], []

                for idx in start_idx:
                    rearr_input.append(mat_mix[:, idx:idx+patch_len, np.newaxis])
                rearr_input = np.array(rearr_input)

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

                timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh[0])
                _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh[1])
                _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh[2])
                _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh[3])

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
                if metrics["Recall"] == 0.0:
                    Fscore = 0

                elif metrics["Precision"] == 0:
                    Fscore = 0

                else:
                    Fscore = 2 * (metrics["Recall"] * metrics["Precision"]) / (metrics["Recall"] + metrics["Precision"])

                metrics["F-Score"] = Fscore
                fscores.append(Fscore)

                metrics["track"] = song
                all_metrics.append(metrics)


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

                timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh[0])
                _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh[1])
                _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh[2])
                _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh[3])

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
                if metrics["Recall"] == 0.0:
                    Fscore = 0

                elif metrics["Precision"] == 0:
                    Fscore = 0

                else:
                    Fscore = 2 * (metrics["Recall"] * metrics["Precision"]) / (metrics["Recall"] + metrics["Precision"])

                metrics["F-Score"] = Fscore
                fscores.append(Fscore)

                metrics["track"] = song
                all_metrics.append(metrics)

            else:
                raise ValueError("Incorrect mode")
        except:
            continue

    pd.DataFrame(all_metrics).to_csv("{}_{}_all_scores.csv".format(exp_name, 'testset'))
    print("Average F-Score in the test set is: {}".format(np.mean(fscores)))



def evaluate_full_individual(test_files, model, patch_len, exp_name, thresh, mode):
    '''Evaluate the model on the test set, full files
    '''

    all_metrics = []
    fscores = []

    ###
    for song in test_files:


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

            # # construct the multi-pitch reference
            #
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

            timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh[0])
            _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh[1])
            _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh[2])
            _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh[3])

            ## multipitch metrics

            est_b = []
            for row in bas: est_b.append(np.array([row]))
            est_t = []
            for row in ten: est_t.append(np.array([row]))
            est_a = []
            for row in alt: est_a.append(np.array([row]))
            est_s = []
            for row in sop: est_s.append(np.array([row]))

            ref_b = []
            for row in ref_freqs_b: ref_b.append(np.array([row]))

            ref_t = []
            for row in ref_freqs_t: ref_t.append(np.array([row]))

            ref_a = []
            for row in ref_freqs_a: ref_a.append(np.array([row]))

            ref_s = []
            for row in ref_freqs_s: ref_s.append(np.array([row]))

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_b, ref_b)):
                if fqs == 0:
                    ref_b[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_t, ref_t)):
                if fqs == 0:
                    ref_t[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_a, ref_a)):
                if fqs == 0:
                    ref_a[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_s, ref_s)):
                if fqs == 0:
                    ref_s[i] = np.array([])


            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_s)):
                if fqs == 0:
                    est_s[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_a)):
                if fqs == 0:
                    est_a[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_t)):
                if fqs == 0:
                    est_t[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_b)):
                if fqs == 0:
                    est_b[i] = np.array([])

            metrics_sop = mir_eval.multipitch.evaluate(ref_times_s, ref_s, timestamp, est_s)
            metrics_alt = mir_eval.multipitch.evaluate(ref_times_a, ref_a, timestamp, est_a)
            metrics_ten = mir_eval.multipitch.evaluate(ref_times_t, ref_t, timestamp, est_t)
            metrics_bass = mir_eval.multipitch.evaluate(ref_times_b, ref_b, timestamp, est_b)


            metrics_sop['F-Score'] = 2 * (metrics_sop["Precision"] * metrics_sop["Recall"]) / (metrics_sop["Precision"] + metrics_sop["Recall"])
            metrics_sop['track'] = song
            metrics_sop['voice'] = "SOPRANO"

            metrics_alt['F-Score'] = 2 * (metrics_alt["Precision"] * metrics_alt["Recall"]) / (metrics_alt["Precision"] + metrics_alt["Recall"])
            metrics_alt['track'] = song
            metrics_alt['voice'] = "ALTO"

            metrics_ten['F-Score'] = 2 * (metrics_ten["Precision"] * metrics_ten["Recall"]) / (metrics_ten["Precision"] + metrics_ten["Recall"])
            metrics_ten['track'] = song
            metrics_ten['voice'] = "TENOR"

            metrics_bass['F-Score'] = 2 * (metrics_bass["Precision"] * metrics_bass["Recall"]) / (metrics_bass["Precision"] + metrics_bass["Recall"])
            metrics_bass['track'] = song
            metrics_bass['voice'] = "BASS"

            all_metrics.append(metrics_sop)
            all_metrics.append(metrics_alt)
            all_metrics.append(metrics_ten)
            all_metrics.append(metrics_bass)

            ## melody metrics
            # metrics_sop = mir_eval.melody.evaluate(ref_times_s, ref_freqs_s, timestamp, sop)
            # metrics_alt = mir_eval.melody.evaluate(ref_times_a, ref_freqs_a, timestamp, alt)
            # metrics_ten = mir_eval.melody.evaluate(ref_times_t, ref_freqs_t, timestamp, ten)
            # metrics_bass = mir_eval.melody.evaluate(ref_times_b, ref_freqs_b, timestamp, bas)
            #
            # metrics_sop['track'] = song
            # metrics_sop['voice'] = "SOPRANO"
            # metrics_alt['track'] = song
            # metrics_alt['voice'] = "ALTO"
            # metrics_ten['track'] = song
            # metrics_ten['voice'] = "TENOR"
            # metrics_bass['track'] = song
            # metrics_bass['voice'] = "BASS"
            #
            # all_metrics.append(metrics_sop)
            # all_metrics.append(metrics_alt)
            # all_metrics.append(metrics_ten)
            # all_metrics.append(metrics_bass)




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

            # # construct the multi-pitch reference
            #
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

            timestamp, sop = utils.pitch_activations_to_mf0(pred_s, thresh=thresh[0])
            _, alt = utils.pitch_activations_to_mf0(pred_a, thresh=thresh[1])
            _, ten = utils.pitch_activations_to_mf0(pred_t, thresh=thresh[2])
            _, bas = utils.pitch_activations_to_mf0(pred_b, thresh=thresh[3])

            ## multipitch metrics

            est_b = []
            for row in bas: est_b.append(np.array([row]))
            est_t = []
            for row in ten: est_t.append(np.array([row]))
            est_a = []
            for row in alt: est_a.append(np.array([row]))
            est_s = []
            for row in sop: est_s.append(np.array([row]))

            ref_b = []
            for row in ref_freqs_b: ref_b.append(np.array([row]))

            ref_t = []
            for row in ref_freqs_t: ref_t.append(np.array([row]))

            ref_a = []
            for row in ref_freqs_a: ref_a.append(np.array([row]))

            ref_s = []
            for row in ref_freqs_s: ref_s.append(np.array([row]))

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_b, ref_b)):
                if fqs == 0:
                    ref_b[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_t, ref_t)):
                if fqs == 0:
                    ref_t[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_a, ref_a)):
                if fqs == 0:
                    ref_a[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(ref_times_s, ref_s)):
                if fqs == 0:
                    ref_s[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_s)):
                if fqs == 0:
                    est_s[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_a)):
                if fqs == 0:
                    est_a[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_t)):
                if fqs == 0:
                    est_t[i] = np.array([])

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(timestamp, est_b)):
                if fqs == 0:
                    est_b[i] = np.array([])

            metrics_sop = mir_eval.multipitch.evaluate(ref_times_s, ref_s, timestamp, est_s)
            metrics_alt = mir_eval.multipitch.evaluate(ref_times_a, ref_a, timestamp, est_a)
            metrics_ten = mir_eval.multipitch.evaluate(ref_times_t, ref_t, timestamp, est_t)
            metrics_bass = mir_eval.multipitch.evaluate(ref_times_b, ref_b, timestamp, est_b)

            metrics_sop['F-Score'] = 2 * (metrics_sop["Precision"] * metrics_sop["Recall"]) / (
                        metrics_sop["Precision"] + metrics_sop["Recall"])
            metrics_sop['track'] = song
            metrics_sop['voice'] = "SOPRANO"

            metrics_alt['F-Score'] = 2 * (metrics_alt["Precision"] * metrics_alt["Recall"]) / (
                        metrics_alt["Precision"] + metrics_alt["Recall"])
            metrics_alt['track'] = song
            metrics_alt['voice'] = "ALTO"

            metrics_ten['F-Score'] = 2 * (metrics_ten["Precision"] * metrics_ten["Recall"]) / (
                        metrics_ten["Precision"] + metrics_ten["Recall"])
            metrics_ten['track'] = song
            metrics_ten['voice'] = "TENOR"

            metrics_bass['F-Score'] = 2 * (metrics_bass["Precision"] * metrics_bass["Recall"]) / (
                        metrics_bass["Precision"] + metrics_bass["Recall"])
            metrics_bass['track'] = song
            metrics_bass['voice'] = "BASS"

            all_metrics.append(metrics_sop)
            all_metrics.append(metrics_alt)
            all_metrics.append(metrics_ten)
            all_metrics.append(metrics_bass)


            ## melody metrics
            # metrics_sop = mir_eval.melody.evaluate(ref_times_s, ref_freqs_s, timestamp, sop)
            # metrics_alt = mir_eval.melody.evaluate(ref_times_a, ref_freqs_a, timestamp, alt)
            # metrics_ten = mir_eval.melody.evaluate(ref_times_t, ref_freqs_t, timestamp, ten)
            # metrics_bass = mir_eval.melody.evaluate(ref_times_b, ref_freqs_b, timestamp, bas)
            #
            # metrics_sop['track'] = song
            # metrics_sop['voice'] = "SOPRANO"
            # metrics_alt['track'] = song
            # metrics_alt['voice'] = "ALTO"
            # metrics_ten['track'] = song
            # metrics_ten['voice'] = "TENOR"
            # metrics_bass['track'] = song
            # metrics_bass['voice'] = "BASS"
            #
            # all_metrics.append(metrics_sop)
            # all_metrics.append(metrics_alt)
            # all_metrics.append(metrics_ten)
            # all_metrics.append(metrics_bass)

        else:
            raise ValueError("Wrong mode")


    pd.DataFrame(all_metrics).to_csv("{}_{}_all_scores_individual_opt_multipitch_oa.csv".format(exp_name, 'testset'))

