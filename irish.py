import tensorflow as tf


start = 0


def get_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28))
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adagrad(0.1)
                  )
    return model


if __name__ == '__main__':
    print('Tensorflow version in {}'.format(tf.__version__))
    model = get_model()
