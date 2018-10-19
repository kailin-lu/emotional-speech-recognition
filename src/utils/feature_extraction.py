# Transform WAV files into
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc, delta

sample_rate, sig = wavfile.read('/Users/kailinlu/Documents/emotional-speech/data/OAF_bite_disgust.wav')

mfcc_feat = mfcc(sig, sample_rate)
delta_feat = delta(mfcc_feat, 1)
delta_delta_feat = delta(delta_feat, 1)
feat = np.concatenate([mfcc_feat, delta_feat, delta_delta_feat], axis=1)
print(feat.shape)
