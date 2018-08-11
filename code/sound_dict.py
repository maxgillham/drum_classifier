
'''
Just a dictionary to correspond wav file names to a number, which is then
one hot encoded using keras.utils to_catagorial.
An inverted mapping is also included for the print satement in keras_classification.py
'''

sound_string_to_num = {
    'snare': 0,
    'hihat': 1,
    'kick': 2
}

sound_num_to_string = {i: j for j, i in sound_string_to_num.items()}
