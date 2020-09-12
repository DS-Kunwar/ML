import absl
import tensorflow as tf
import kerastuner


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_hyperparam() -> kerastuner.HyperParameters:

    hp = kerastuner.HyperParameters()
    hp.Choice(name='learning_rate', values=[1e-2, 1e-3], default=1e-2)
    return hp


def _build_model(hp) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    return model


tuner = kerastuner.Hyperband(
    _build_model,
    objective='accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    hyperparameters=get_hyperparam(),
    project_name='Fasion_tuner',
)


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)

    tuner.search(train_images, train_labels, epochs=10)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_images, train_labels, epochs=10)
