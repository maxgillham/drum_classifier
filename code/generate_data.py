import numpy as np
import librosa
import os

from sound_dict import sound_string_to_num

'''
This is the method to call to return mfcc data of training samples.   Training samples must be
formatted as snare_1.wav, kick_3.wav, hihat_2.wav ect...  This will return a three dimensional array
where the first index is the batch index and the seccond and third are the mfcc array
'''
def form_training_data():
    os.chdir(r"C:\Users\maxfg\Desktop\beatBox\datastore\training")
    sound_mfcc, labels, max_length = get_data_for_path()
    sound_mfcc = pad_mfcc(mfcc_list=sound_mfcc,max_length=max_length)
    return sound_mfcc, labels, max_length

'''
Same as above but for testing data
'''
def form_testing_data(max_length):
    os.chdir(r"C:\Users\maxfg\Desktop\beatBox\datastore\testing")
    sound_mfcc, labels, max_length_for_test = get_data_for_path()
    sound_mfcc = pad_mfcc(mfcc_list=sound_mfcc,max_length=max_length)
    return sound_mfcc, labels

'''
This method iterates through whatever training files are in the folder specified in form_training_data()
and returns the computed mfccs, their corresponding numeric labels (see dictionary) and the maximum length
of training data for zero padding
'''
def get_data_for_path():
    mfcc_list = []
    lengths = []
    labels = []
    files = os.listdir(os.getcwd())
    for i in range(len(files)):
        mfcc, length = convert_to_mfcc(file_path=files[i])
        mfcc_list.append(mfcc)
        lengths.append(length)
        labels.append(sound_string_to_num[files[i].split('_')[0]])
    return mfcc_list, labels, max(lengths)

'''
This method actually computes the mfcc of each wave file given a path
'''
def convert_to_mfcc(file_path):
    wav, sample_rate = librosa.load(file_path, mono=True, sr=None)
    wav = wav[::3]
    mfcc = librosa.feature.mfcc(wav, sr=16000)
    return mfcc, mfcc.shape[1]

'''
This method pads each mfcc to a specified length to ensure tensors are finite dimensional
'''
def pad_mfcc(mfcc_list, max_length):
    for i in range(len(mfcc_list)):
        pad_width = max_length - mfcc_list[i].shape[1]
        mfcc_list[i] = np.pad(mfcc_list[i], pad_width=((0,0),(0,pad_width)), mode='constant')
    return np.array(mfcc_list)




