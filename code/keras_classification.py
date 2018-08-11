import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import to_categorical

from generate_data import form_training_data, form_testing_data
from sound_dict import sound_num_to_string

'''
This is used to to train a simple classification model using keras.  You will pass
it the mfcc data and corresponding vectorized labels
'''
def train_model(x_data,y_data):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(2,2), activation='softmax', data_format='channels_last', input_shape=(x_data.shape[1],x_data.shape[2],1)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    ada = Adadelta(lr=1e-1)

    model.compile(loss='categorical_crossentropy', optimizer=ada)
    history = model.fit(x_data,y_data,epochs=500, batch_size=1, verbose=2)
    plt.plot(history.history['loss'])
    plt.show()
    model.save(r"C:\Users\maxfg\Desktop\drum_classifier\datastore\keras_models\new_model.h5")
    del model
    return

'''
This method returns predictions given a set of inputs based off of a saved model
'''
def model_predict(x_data):
    model = load_model(r"C:\Users\maxfg\Desktop\drum_classifier\datastore\keras_models\new_model.h5")
    out = model.predict(x_data)
    return out



def begin_process():
    x_data, y_data, pad_length = form_training_data()
    #have to add channel to fit model shape and one hot encode output for crossentropy
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    y_data = to_categorical(y_data)
    train_model(x_data=x_data, y_data=y_data)

    test_data, correct_outputs = form_testing_data(pad_length)
    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
    output = model_predict(test_data)
    for i in range(output.shape[0]):
        print('\nSnare guess: ', np.round(output[i, 0], 3), 'Hihat guess: ', np.round(output[i, 1], 3), 'Kick guess: ', np.round(output[i, 2],3))
        print(sound_num_to_string[correct_outputs[i]], ' is the correct answer')
    return

if __name__ == '__main__':
    begin_process()