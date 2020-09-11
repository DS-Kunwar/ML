import os
import pandas as pd 
import tensorflow as tf 

def get_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28,28))
    ])

    return model

if __name__ == '__main__':
    print('Tensorflow version in {}'.format(tf.__version__))
    model = get_model()
