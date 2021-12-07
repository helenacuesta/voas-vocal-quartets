'''Generator function for training and validation of the models.
Code adapted from Pritish' voas.
'''


import numpy as np

import random
import csv

import os
import glob

import voas.config as config
import voas.utils as utils

import pandas as pd

import pescador
import unidecode



def full_patch_generator_freq(song, max_phr_len):

    mat_mix = []
    with open(os.path.join(
            config.feats_dir, "{}_mix.csv".format(song)
        ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_mix.append(np.float32(line))
    mat_mix = np.array(mat_mix)

    mat_S = []
    with open(os.path.join(
            config.feats_dir, "{}_S.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_S.append(np.float32(line))
    mat_S = np.array(mat_S)

    mat_A = []
    with open(os.path.join(
            config.feats_dir, "{}_A.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_A.append(np.float32(line))
    mat_A = np.array(mat_A)

    mat_T = []
    with open(os.path.join(
            config.feats_dir, "{}_T.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_T.append(np.float32(line))
    mat_T = np.array(mat_T)

    mat_B = []
    with open(os.path.join(
            config.feats_dir, "{}_B.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_B.append(np.float32(line))
    mat_B = np.array(mat_B)

    time_len = mat_mix.shape[1]

    if time_len <= max_phr_len:
        pass

    start_idxs = np.arange(0, time_len - max_phr_len, step=max_phr_len)
    start_idxs = np.random.permutation(start_idxs)

    for idx in start_idxs:

        idx_end = int(idx + max_phr_len)

        yield dict(
            X=mat_mix[:, idx:idx_end][:, :, np.newaxis],
            Y1=mat_S[:, idx:idx_end],
            Y2=mat_A[:, idx:idx_end],
            Y3=mat_T[:, idx:idx_end],
            Y4=mat_B[:, idx:idx_end]
        )

def full_patch_generator_time(song, max_phr_len):

    song = song.decode("utf8")
    mat_mix = []
    with open(os.path.join(config.feats_dir, "{}_mix.csv".format(song)
        ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_mix.append(np.float32(line))
    mat_mix = np.array(mat_mix)

    mat_S = []
    with open(os.path.join(
            config.feats_dir, "{}_S.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_S.append(np.float32(line))
    mat_S = np.array(mat_S)

    mat_A = []
    with open(os.path.join(
            config.feats_dir, "{}_A.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_A.append(np.float32(line))
    mat_A = np.array(mat_A)

    mat_T = []
    with open(os.path.join(
            config.feats_dir, "{}_T.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_T.append(np.float32(line))
    mat_T = np.array(mat_T)

    mat_B = []
    with open(os.path.join(
            config.feats_dir, "{}_B.csv".format(song)
    ), "r") as f:
        rd = csv.reader(f)
        for line in rd: mat_B.append(np.float32(line))
    mat_B = np.array(mat_B)

    time_len = mat_mix.shape[1]

    if time_len <= max_phr_len:
        pass

    # we want consecutive patches for "sorted training"
    start_idxs = np.arange(0, time_len - max_phr_len, step=max_phr_len)
    #start_idxs = np.random.permutation(start_idxs)

    for idx in start_idxs:

        idx_end = int(idx + max_phr_len)

        yield dict(
            X=mat_mix[:, idx:idx_end].transpose()[:, :, np.newaxis],
            Y1=mat_S[:, idx:idx_end].transpose(),
            Y2=mat_A[:, idx:idx_end].transpose(),
            Y3=mat_T[:, idx:idx_end].transpose(),
            Y4=mat_B[:, idx:idx_end].transpose()
        )


def full_generator_pescador(data_list, mode, patch_len, batch_size=config.batch_size):
    """Data generator using pescador
    """

    mode = mode.decode("utf-8")

    if mode == "time":

        streams = [
            pescador.Streamer(full_patch_generator_time, song, patch_len) for song in data_list
        ]
        mux_stream = pescador.StochasticMux(streams, n_active=5, rate=None, mode="single_active")

    elif mode == "freq":

        streams = [
                pescador.Streamer(full_patch_generator_freq, song.decode("utf-8"), patch_len) for song in data_list
            ]
        mux_stream = pescador.StochasticMux(streams, n_active=5, rate=None, mode="single_active")


    else:
        raise ValueError("Wrong mode. Expected 'time' or 'freq' and found {}".format(mode))


    batch_generator = pescador.buffer_stream(mux_stream, batch_size)

    for batch in batch_generator:
        yield (batch['X'], (batch['Y1'], batch['Y2'], batch['Y3'], batch['Y4']))
