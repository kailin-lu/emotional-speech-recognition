# Emotional Speech Recognition 

This is a 2 layer bi-directional LSTM model which classifies 7 emotions from an audio 
file: 

* Happy 
* Sad 
* Pleasant Surprise 
* Anger 
* Neutral 
* Fear
* Disgust 

### Data 

The training data is from the [Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487) 
Two actresses aged 26 and 64 read a variety of sentences structured in the format

'Say the word _____' using different words. 

There are a total of 2800 audio files with an equal distribution across all classes. 

Each audio file was preprocessed using [python_speech_features](https://github.com/jameslyons/python_speech_features)
into mel-frequency cepstral coefficients (MFCC), delta (first difference) and delta-delta (first difference of differences) features, 
totaling 39 features per 25ms segment of audio. 

### Results 

Accuracy (one vs. all): 93% 
F1 Score: 93% 
