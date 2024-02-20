# 1. The problem of vanishing and exploding gradients

# 2. No, you need to break the symmetry for neurons to take different values

# 3. Yes

# 4. Use relu for shallow networks, swish for deep. For a self-normalizing net use selu.

# 5. If a very large value is used there's little friction so the learning rate can grow too large and overshoot the
# optimum.

# 6. You can produce a sparse model with strong l1 regularization, the tensorflow optimization toolkit or just turning
# off neurons with small weights.

# 7. Dropout does slow down training as it makes models converge slower. It doesn't slow down inference. MC dropout
# slows down inference.
from functools import partial
from pathlib import Path

import keras_tuner
import numpy
import tensorflow
from sklearn.model_selection import train_test_split

from at_exercises.e10 import get_run_logdir


def get_data(test=False):
    (data, labels), (test_data, test_labels) = tensorflow.keras.datasets.cifar10.load_data()
    data = data / 255.
    test_data = test_data / 255.
    if test:
        return train_test_split(data, labels, random_state=0), (test_data, test_labels)
    else:
        return train_test_split(data, labels, random_state=0)


def get_callbacks(tb_path=Path("logs/cifar10/my_rnd_search/tensorboard"), patience=2):
    tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(tb_path, profile_batch=(100, 200))
    early_stopping_cb = tensorflow.keras.callbacks.EarlyStopping(patience=patience)
    return [tensorboard_cb, early_stopping_cb]


def q8b_learning_rate_search():
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks()
    random_search_tuner = keras_tuner.RandomSearch(
        build_model_for_search,
        objective="val_accuracy",
        max_trials=50,
        overwrite=True,
        directory="logs/cifar10",
        project_name="my_rnd_search",
        seed=0
    )
    random_search_tuner.search(
        data,
        labels,
        epochs=10,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )


def q8b_train():
    learning_rate = 0.0001
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8_train/tensorboard"), patience=10)
    model = build_model(learning_rate)
    history = model.fit(
        data,
        labels,
        epochs=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )
    return history


def q8c_learning_rate_search():
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8c_learning_rate_search/tensorboard"))
    random_search_tuner = keras_tuner.RandomSearch(
        partial(build_model_for_search, batch_normalization=True),
        objective="val_accuracy",
        max_trials=50,
        overwrite=True,
        directory="logs/cifar10",
        project_name="q8c_learning_rate_search",
        seed=0
    )
    random_search_tuner.search(
        data,
        labels,
        epochs=10,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )


def q8c_train():
    learning_rate = 0.0001
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8c_train/tensorboard"), patience=10)
    model = build_model(learning_rate, batch_normalization=True)
    history = model.fit(
        data,
        labels,
        epochs=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )
    return history


def q8d_learning_rate_search():
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8d_learning_rate_search/tensorboard"))
    random_search_tuner = keras_tuner.RandomSearch(
        build_selu_model_for_search,
        objective="val_accuracy",
        max_trials=50,
        overwrite=True,
        directory="logs/cifar10",
        project_name="q8d_learning_rate_search",
        seed=0
    )
    random_search_tuner.search(
        data,
        labels,
        epochs=10,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )


def q8d_train():
    learning_rate = 0.001
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8d_train/tensorboard"), patience=10)
    model = build_selu_model(learning_rate)
    history = model.fit(
        data,
        labels,
        epochs=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )
    return history


def q8e_learning_rate_search():
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8e_learning_rate_search/tensorboard"))
    random_search_tuner = keras_tuner.RandomSearch(
        partial(build_selu_model_for_search, dropout=True),
        objective="val_accuracy",
        max_trials=50,
        overwrite=True,
        directory="logs/cifar10",
        project_name="q8e_learning_rate_search",
        seed=0
    )
    random_search_tuner.search(
        data,
        labels,
        epochs=10,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )


def q8e_train():
    learning_rate = 0.001
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8e_train/tensorboard"), patience=100)
    model = build_selu_model(learning_rate, dropout=True)
    history = model.fit(
        data,
        labels,
        epochs=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )
    return history

def q8f_learning_rate_search():
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8f_learning_rate_search/tensorboard"))
    random_search_tuner = keras_tuner.RandomSearch(
        partial(build_selu_model_for_search, dropout=True),
        objective="val_accuracy",
        max_trials=50,
        overwrite=True,
        directory="logs/cifar10",
        project_name="q8f_learning_rate_search",
        seed=0
    )
    random_search_tuner.search(
        data,
        labels,
        epochs=10,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )


def q8f_train():
    learning_rate = 0.001
    data, validation_data, labels, validation_labels = get_data()
    callbacks = get_callbacks(tb_path=Path("logs/cifar10/q8f_train/tensorboard"), patience=100)
    model = build_selu_model(learning_rate, dropout=True)
    history = model.fit(
        data,
        labels,
        epochs=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks
    )
    return history


def mc_dropout(model):
    (data, validation_data, labels, validation_labels), (test_data, test_labels) = get_data(test=True)
    y_probas = numpy.stack([model(test_data, training=True) for _ in range(100)])
    y_proba = y_probas.mean(axis=0)
    y_pred = y_proba.argmax(axis=1)
    return (y_pred == test_labels.reshape(1, -1)).sum() / len(test_labels)


def build_model_for_search(hp, batch_normalization=False):
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    return build_model(learning_rate, batch_normalization=batch_normalization)


def build_model(learning_rate, batch_normalization=False):
    optimizer = tensorflow.keras.optimizers.legacy.Nadam(learning_rate=learning_rate)
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=[32, 32, 3]),
    ])
    for _ in range(10):
        model.add(tensorflow.keras.layers.Dense(100, activation="swish", kernel_initializer="he_normal"))
        if batch_normalization:
            model.add(tensorflow.keras.layers.BatchNormalization())
    model.add(tensorflow.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def build_selu_model_for_search(hp, dropout=False):
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    return build_selu_model(learning_rate, dropout=dropout)


def build_selu_model(learning_rate, dropout=False):
    optimizer = tensorflow.keras.optimizers.legacy.Nadam(learning_rate=learning_rate)
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=[32, 32, 3]),
        tensorflow.keras.layers.Normalization()
    ])
    for _ in range(10):
        model.add(tensorflow.keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
        if dropout:
            model.add(tensorflow.keras.layers.AlphaDropout(0.2))
    model.add(tensorflow.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


