import os

from voas import config, utils
import numpy as np


from mir_eval.util import midi_to_hz, intervals_to_samples
import mir_eval
import pandas as pd
import pretty_midi

from scipy.signal import medfilt



def midi_preparation(midifile):

    midi_data = dict()
    midi_data['onsets'] = dict()
    midi_data['offsets'] = dict()
    midi_data['midipitches'] = dict()  # midi notes?
    midi_data['hz'] = dict()

    patt = pretty_midi.PrettyMIDI(midifile)

    if len(patt.instruments) != 4:
        print("Parsing only the first 4 voices of {}.".format(midifile))
        # if less than four voices, we skip
        if len(patt.instruments) < 4:
            return None

    midi_data['downbeats'] = patt.get_downbeats()

    voices = ['S', 'A', 'T', 'B']

    idx=-1
    average_pitches = []
    for i in range(4):
        instrument = patt.instruments[i]

        idx += 1

        midi_data['onsets'][voices[idx]] = []
        midi_data['offsets'][voices[idx]] = []
        midi_data['midipitches'][voices[idx]] = []

        for note in instrument.notes:
            midi_data['onsets'][voices[idx]].append(note.start)
            midi_data['offsets'][voices[idx]].append(note.end)
            midi_data['midipitches'][voices[idx]].append(note.pitch)

        p = midi_data['midipitches'][voices[idx]]
        midi_data['hz'][voices[idx]] = midi_to_hz(np.array(p))

        average_pitches.append(np.mean(midi_data['hz'][voices[idx]]))

    if np.sum(np.abs(np.array(average_pitches, dtype=np.float32) - np.array(np.sort(average_pitches)[::-1], dtype=np.float32))) > 0:
        print("Voices do not seem to be sorted as SATB in the score, skipping {}".format(midifile))
        return None

    else:
        return midi_data

def midi_to_trajectory(onsets, offsets, pitches, hop=config.hopsize/config.fs):

    """Convert the MIDI notes to the associated time series representation, frame-wise.
    Frame size determined by hopsize/samplingRate
    """

    intervals = np.concatenate([np.array(onsets)[:, None], np.array(offsets)[:, None]], axis=1)
    timebase, midipitches = intervals_to_samples(intervals, list(pitches),
                                                 sample_size=hop, fill_value=0)

    return np.array(timebase), np.array(midipitches)

"""Main code for data preparation. Read scores (XML to MIDI conversion when necessary), generate input/output feature pairs.
"""

## restricted to SATB
voices = ['S', 'A', 'T', 'B']

for id_score in os.listdir(config.scores_dir):

    if os.path.exists(os.path.join(config.feats_dir, "{}_mix.csv".format(id_score))):
        continue

    score_format = id_score.split('.')[-1]  # indicates the extension of the score file
    id_score = id_score.split('.')[0]


    if score_format == 'mid':

        print("Parsing {}.mid MIDI file...".format(id_score, score_format))

        midi_data = midi_preparation(os.path.join(config.scores_dir, "{}.mid".format(id_score)))

        if midi_data is None: continue

    else:
        print("Trying to convert {}.{} to MIDI...".format(id_score, score_format))


        out = utils.xml2midi(
            "{}.{}".format(id_score, score_format),
            score_format
        )
        if out is None: continue

        midi_data = midi_preparation(os.path.join(config.midis_dir, "{}.mid".format(id_score)))

        if midi_data is None: continue
    max_len = 0
    timebases, f0 = [], []
    for idx, voice in enumerate(voices):

        onsets = midi_data['onsets'][voice]
        offsets = midi_data['offsets'][voice]
        pitches = midi_data['hz'][voice]
        timebase, f0s = midi_to_trajectory(onsets, offsets, pitches)
        timebases.append(timebase)
        f0.append(f0s)
        if len(timebase) >= max_len:
            max_len = len(timebase)
            arg_max_len = idx

    times = timebases[arg_max_len]  # timebase to resample all voices

    # time and freq grid are fixed for the whole song
    time_grid = utils.get_time_grid(len(times))
    freq_grid = utils.get_freq_grid()

    mixture_salience = 0
    for idx, voice in enumerate(voices):

        freqs, voicing = mir_eval.melody.freq_to_voicing(f0[idx])
        freqs, _ = mir_eval.melody.resample_melody_series(timebases[idx], freqs, voicing, times, kind='nearest')

        ## f0 contour degradation
        for f in range(len(freqs)):
            if freqs[f] == 0: continue
            freqs[f] += np.random.normal(scale=5)

        freqs = medfilt(freqs, kernel_size=7)

        synth_salience = utils.create_annotation_target(freq_grid, time_grid, times, freqs)
        # np.save(os.path.join(config.feats_dir, "{}_{}.npy".format(id_score, voice)), synth_salience)

        pd.DataFrame(synth_salience).to_csv(os.path.join(config.feats_dir, "{}_{}.csv".format(id_score, voice)),
                                            header=None, index=False)

        mixture_salience += synth_salience

    mixture_salience[mixture_salience > 1] = 1
    pd.DataFrame(mixture_salience).to_csv(os.path.join(config.feats_dir, "{}_mix.csv".format(id_score)),
                                        header=None, index=False)


    print("Input/output features for {} are successfully exported.".format(id_score))











