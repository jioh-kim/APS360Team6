import deeplake
import librosa
import numpy as np
import resampy
import random

from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# Load the data (training)
ds = deeplake.load("hub://activeloop/nsynth-train")

def load_audio(dataset):
    # Filter the data to only include piano
    filtered_ds = []
    for i in range(1000):
        instrument_family = dataset[i].instrument_family.data()['text'][0]
        if instrument_family == 'keyboard':
             filtered_ds.append(ds[i].audios.data())
    
    return filtered_ds 


# Audio preprocessing, will be inputting each index of load_audio() as a parameter
def audio_preprocessing(a):
    # Convert audios to .wav
    audio = a.numpy()
    # Convert audio to CQT (Constant-Q Transform)
    audio = librosa.cqt(y, sr=16000)
    # Trim the audio
    audio = audio[:int(3 * 16000)]

# Adjust the MIDI labels to be in the range [0, 88]
def adjust_labels(midi_value):
    return midi_value - 21

load_audio(ds)
for audio in load_audio(ds):
    audio_preprocessing(audio)

ds.summary()



        

