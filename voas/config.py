import os

scores_dir = './data/rawScores'
feats_dir = './data/feats'
midis_dir = './data/midis'
plots_dir = './data/plots'
models_dir = './models'
# scores_dir = './data/rawScores'
# feats_dir = './data/feats'
# midis_dir = './data/midis'
# plots_dir = './data/plots'
# models_dir = './models'

# scores_dir = '/Users/helenacuesta/Desktop/MTG/datasets/VoiceAssignmentScores'
# feats_dir = '/Users/helenacuesta/Desktop/MTG/datasets/VoiceAssignmentScores/feats'
# midis_dir = '/Users/helenacuesta/Desktop/MTG/datasets/VoiceAssignmentScores/midis'
# plots_dir = '/Users/helenacuesta/Desktop/MTG/datasets/VoiceAssignmentScores/plots'


fs = 22050.0
hopsize = 256
framesize = 1024
bins_per_octave = 60
n_octaves = 6
over_sample = 5
harmonics = 1, 2, 3, 4, 5
fmin = 32.7

batch_size = 16
train_split = 0.9
num_features = 360
init_lr = 0.0005
max_models_to_keep = 3
print_every = 1
save_every = 10
lstm_size = 32
samples_per_file = 4
blur = True
blur_max_std = 1
num_epochs = 100
batches_per_epoch = 4096
validation_steps = 1024

def max_phr_len(patch_len):
    return patch_len


'''Generate song list and data partitions
'''
file_list = os.listdir(feats_dir)
songs = [song[:-8] for song in file_list if song.endswith('_mix.csv')]






