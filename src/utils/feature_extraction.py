# Data Pre-processing

import os
from collections import defaultdict
import pickle
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc, delta


def transform_mfcc(wavefile):
    """Transforms wav file in to [x, 39] array of mfcc,
    delta, and delta-delta features

    :param wavefile: wav file
    """
    try:
        sample_rate, sig = wavfile.read(wavefile)
    except ValueError:
        print(wavefile)
    mfcc_feat = mfcc(sig, sample_rate)
    delta_feat = delta(mfcc_feat, 1)
    delta_delta_feat = delta(delta_feat, 1)
    return np.concatenate([mfcc_feat, delta_feat, delta_delta_feat], axis=1)


def load_labels(data='/Users/kailinlu/Documents/emotional-speech/data/'):
    """Returns one hot encoded labels

    :return: one hot labels array, int_labels, labels to int mapping
    """
    files = os.listdir(data)
    labels = [file.split('_')[2].split('.')[0] for file in files]
    labels_map = defaultdict(int)
    i = 0
    for label in list(set(labels)):
        labels_map[label] = i
        i += 1
    int_labels = np.array([labels_map[label] for label in labels])
    onehot = np.zeros((len(int_labels), max(int_labels) + 1))
    onehot[np.arange(len(int_labels)), int_labels] = 1
    return onehot, int_labels, labels_map


def pad_sequence(sequence, max_length):
    """Pad sequence to the left with 0s
    sequence [?, x] to [max_length, x] with 0"""

    dim_sequence = sequence.shape

    if dim_sequence[0] > max_length:
        return sequence[:max_length, :], max_length

    elif dim_sequence[0] == max_length:
        return sequence, max_length
    else:
        pad_size = max_length - dim_sequence[0]
        pad = np.zeros((pad_size, dim_sequence[1]))
        return np.concatenate((sequence, pad), axis=0), dim_sequence[0]


def load_audio(data='/Users/kailinlu/Documents/emotional-speech/data/'):
    """Load audio data into numpy array
    """
    files = os.listdir(data)
    audio = [transform_mfcc(data + file) for file in files]
    max_length = max([a.shape[0] for a in audio])
    padded_audio = [pad_sequence(a, max_length) for a in audio]
    audio = [a[0][np.newaxis, :,:] for a in padded_audio]
    lengths = [a[1] for a in padded_audio]
    return np.concatenate(audio, axis=0), lengths


if __name__=='__main__':
    audio = load_audio()
    with open('/Users/kailinlu/Documents/emotional-speech/pickle/data.p', 'wb') as f:
        pickle.dump(audio, f)

    labels = load_labels()
    with open('/Users/kailinlu/Documents/emotional-speech/pickle/labels.p', 'wb') as f:
        pickle.dump(labels, f)