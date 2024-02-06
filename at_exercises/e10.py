# 3. It's generally preferable to use a logistic regression classifier rather than a classic perceptron because a
# logistic regression classifier can provide probabilities as well as classification. Logistic regression classifiers
# are regularized so they tend to generalize better. A perceptron can be made equivalent to a logistic regression
# classifier by swapping the step function for a logistic function.

# 4. Sigmoid ativation function was key to training the first MLPs because it has a well defined gradient everywhere for
# gradient descent.

# 5. 3 popular activation functions are rectified linear, sigmoid and tanh.

# 6a. The input matrix will be m x 10.

# 6b. W_h will be 10x50. b_h will be a 50D vector.

# 6c. W_o will be 50x3. b_o will be a 3D vector.

# 6d. Y will be a 3D vector.

# 6e. Y = phi(phi(X @ W_h + b_h) @ W_o _ b_o), where phi is RELU

# 7. 1 Neuron is required to classify email as spam or ham. In the output layer use sigmoid. To tackle mnist, which
# has 10 digits you need 10 output neurons. Use softmax in the output layer. For predicting housing prices a single
# output neuron is required, use ReLU or softplus to guarantee the output is non-negative.

# 8. Backpropogation is a combination of gradient descent and reverse-mode autodiff. It works on one mini-batch at a
# time, and goes through the full training set multiple times (each pass is called an epoch). First is the forward pass,
# each mini-batch is passed through the network, calculating the output of each neuron. The network's error is
# measured using the output. Next the chain rule is used to calculate the contribution of each connection's weight and
# bias to the output error. Then the contribution to each output from the layer below is calculated, and so on down
# through the network until it is calculated for every connection. Finally, each weight and bias is tweaked using the
# error gradient to perform gradient descent.

# 9. In a basic NLP one can tweak, the number of neurons in each layer, the number of layers, the activation function,
# the output activation function, the learning rate, the loss function. If overfitting is happening, reduce the number
# of neurons in each layer, reduce the number of layers.
from pathlib import Path
from time import strftime

import keras_tuner
import tensorflow
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

k = tensorflow.keras.backend


def get_run_logdir(root_logdir="logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")


class ExponentialLearningRate(tensorflow.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(k.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        k.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)


def q10_exponential_learning():
    (data, labels), (test_data, test_labels) = tensorflow.keras.datasets.mnist.load_data()
    data = data / 255.
    test_data = test_data / 255.
    data, validation_data, labels, validation_labels = train_test_split(data, labels, random_state=0)
    tensorflow.random.set_seed(0)
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=[28, 28]),
        tensorflow.keras.layers.Dense(300, activation="relu"),
        tensorflow.keras.layers.Dense(100, activation="relu"),
        tensorflow.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.1),
        metrics=["accuracy"]
    )
    # elr = ExponentialLearningRate(factor=1.005)
    run_logdir = get_run_logdir()  # e.g., my_logs/run_2022_08_01_17_25_59
    tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(run_logdir, profile_batch=(100, 200))
    history = model.fit(
        data,
        labels,
        epochs=30,
        validation_data=(validation_data, validation_labels),
        # callbacks=[tensorboard_cb, elr],
        callbacks=[tensorboard_cb]
    )


def plot_loss_learning_rate(elr):
    pyplot.plot(elr.rates, elr.losses)
    pyplot.gca().set_xscale('log')
    pyplot.hlines(min(elr.losses), min(elr.rates), max(elr.rates))
    pyplot.axis([min(elr.rates), max(elr.rates), 0, elr.losses[0]])
    pyplot.grid()
    pyplot.xlabel("Learning rate")
    pyplot.ylabel("Loss")


def q10_keras_tuner():
    (full_data, full_labels), (test_data, test_labels) = tensorflow.keras.datasets.mnist.load_data()
    full_data = full_data / 255.
    test_data = test_data / 255.
    data, validation_data, labels, validation_labels = train_test_split(full_data, full_labels, random_state=0)
    tensorflow.random.set_seed(0)

    tuner = keras_tuner.Hyperband(
        MyClassificationHyperModel(),
        objective="val_accuracy",
        seed=0,
        max_epochs=10,
        factor=3,
        hyperband_iterations=2,
        overwrite=True,
        directory="logs/my_mnist",
        project_name="hyperband"
    )

    root_logdir = Path(tuner.project_dir) / "tensorboard"
    tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(root_logdir)
    early_stopping_cb = tensorflow.keras.callbacks.EarlyStopping(patience=2)
    tuner.search(
        data,
        labels,
        epochs=10,
        validation_data=(validation_data, validation_labels),
        callbacks=[early_stopping_cb, tensorboard_cb]
    )
    top3_models = tuner.get_best_models(num_models=3)
    best_model = top3_models[0]
    best_model.fit(full_data, full_labels, epochs=10)
    test_loss, test_accuracy = best_model.evaluate(test_data, test_labels)
    print(f"test_loss: {test_loss}")
    print(f"test_accuracy: {test_accuracy}")


def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tensorflow.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    else:
        optimizer = tensorflow.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tensorflow.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tensorflow.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


class MyClassificationHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tensorflow.keras.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, y, **kwargs)
