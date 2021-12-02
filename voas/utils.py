import librosa

import os
import json

import numpy as np

import scipy
from scipy.ndimage import filters
import music21 as m21

import os
import sys

import voas.config as config

import tensorflow as tf
from tensorflow.keras import backend as K



'''Accepted score formats: MIDI, xml, musicxml, mxl
'''

def xml2midi(xmlfile, format):

    try:
        if format == 'mxl':
            score = m21.converter.parseFile(os.path.join(config.scores_dir, xmlfile), format='musicxml')
        else:
            score = m21.converter.parseFile(os.path.join(config.scores_dir, xmlfile), format=format)

    except:
        print("Score {} cannot be parsed by default Music21 parser.".format(xmlfile))
        return None

    try:
        score.write("midi", os.path.join(config.midis_dir, "{}.mid".format(xmlfile.split('.')[0])))
        return "Success"

    except:
        print("Could not convert score {} to MIDI. Skipping...".format(xmlfile))
        return None


def save_json_data(data, save_path):
    with open(save_path, 'w') as fp:
        json.dump(data, fp)


def load_json_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data


def get_hcqt_params():

    bins_per_octave = 60
    n_octaves = 6
    over_sample = 5
    harmonics = [1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256

    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length, over_sample



def get_freq_grid():
    """Get the hcqt frequency grid
    """
    (bins_per_octave, n_octaves, _, _, f_min, _, over_sample) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(
        n_octaves * 12 * over_sample, f_min, bins_per_octave=bins_per_octave)
    return freq_grid


def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    """
    (_, _, _, sr, _, hop_length, _) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length
    )
    return time_grid


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins



def create_annotation_target(freq_grid, time_grid, annotation_times, annotation_freqs):
    """Create the binary annotation target labels with Gaussian blur
    """
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target = np.zeros((n_freqs, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1


    annotation_target_blur = filters.gaussian_filter1d(
        annotation_target, 1, axis=0, mode='constant'
    )
    if len(annot_freq_idx) > 0:
        min_target = np.min(
            annotation_target_blur[annot_freq_idx, annot_time_idx]
        )
    else:
        min_target = 1.0

    annotation_target_blur = annotation_target_blur / min_target
    annotation_target_blur[annotation_target_blur > 1.0] = 1.0

    return annotation_target_blur


def create_data_split(list_of_songs, output_path):


    Ntracks = len(list_of_songs)


    train_perc = 0.75
    validation_perc = 0.1
    test_perc = 1 - train_perc - validation_perc

    # consider doing the training taking into account the songs
    # maybe leaving one song out for evaluation

    songs_randomized = np.random.permutation(list_of_songs)

    train_set = songs_randomized[:int(train_perc * Ntracks)]
    validation_set = songs_randomized[int(train_perc * Ntracks):int(train_perc * Ntracks) + int(validation_perc * Ntracks)]
    test_set = songs_randomized[int(train_perc * Ntracks) + int(validation_perc * Ntracks):]

    data_splits = {
        'train': list(train_set),
        'validate': list(validation_set),
        'test': list(test_set)
    }

    with open(output_path, 'w') as fhandle:
        fhandle.write(json.dumps(data_splits, indent=2))

    return data_splits


def progress(count, total, suffix=''):
    """
    Function to diplay progress bar
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))

    sys.stdout.flush()


def grab_input_slices(input_mat, patch_len):

    '''Input mat will be [num_features x patch_len]
    But the input chunks should be .transpose() of the original
    '''

    max_phr_len = config.max_phr_len(patch_len)

    slice_start_times = np.arange(start=0, stop=input_mat.shape[-1], step=max_phr_len)

    batches = []

    for i in slice_start_times[:-1]:
        chunk = input_mat[:, i:i+max_phr_len]
        batches.append(chunk.transpose())

    last_chunk = np.zeros([config.num_features, max_phr_len])
    last_chunk[:, :input_mat[:, slice_start_times[-1]:].shape[-1]] = input_mat[:, slice_start_times[-1]:]
    batches.append(last_chunk.transpose())

    return batches


def eval_generator(data_batches, patch_len):
    for batch in data_batches:
        yield batch.reshape(1, config.max_phr_len(patch_len), config.num_features)


def bkld(y_true, y_pred):
    """Brian's KL Divergence implementation
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert pitch activation map to multipitch
    by peak picking and thresholding
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    #est_freqs = [[] for _ in range(len(times))]
    est_freqs = np.zeros([len(times), 1])

    for f, t in zip(idx[0], idx[1]):

        if np.array(f).ndim > 1:
            idx_max = peak_thresh_mat[t, f].argmax()
            est_freqs[t] = freqs[f[idx]]

        else:
            est_freqs[t] = freqs[f]


    # est_freqs = [np.array(lst) for lst in est_freqs]

    return times.reshape(len(times),), est_freqs.reshape(len(est_freqs),)

def get_single_chunk_prediction(model, input_mix_mat):

    # for now we only deal with the scenario where we have pre-computed features
    # the input should be already a chunk in this function

    #print("[MSG] >>>>> Input shape for prediction is: {}".format(input_mix_mat.shape))

    p = model.predict(input_mix_mat, verbose=1)

    output_predictions = {}
    output_predictions['sop'] = np.array(p[0])
    output_predictions['alt'] = np.array(p[1])
    output_predictions['ten'] = np.array(p[2])
    output_predictions['bas'] = np.array(p[3])

    return output_predictions