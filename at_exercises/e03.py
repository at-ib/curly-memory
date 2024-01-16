import pickle
import tarfile
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot
from scipy.ndimage import shift
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_mnist_data():
    with open("datasets/mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return mnist


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    pyplot.imshow(image, cmap="binary")
    pyplot.axis("off")


def get_train_test_data(mnist):
    x = mnist.data
    y = mnist.target
    return x[:60000], x[60000:], y[:60000], y[60000:]


def q1():
    mnist = get_mnist_data()
    data, test_data, labels, test_labels = get_train_test_data(mnist)
    knn_clf = Pipeline([("knn", KNeighborsClassifier())])
    param_grid = [
        {
            'knn__weights': ["uniform", "distance"],
            "knn__n_neighbors": [1, 2, 3, 4],
         },
    ]
    search = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy')
    search.fit(data, labels)
    return pandas.DataFrame(search.cv_results_)


def q1_test():
    mnist = get_mnist_data()
    data, test_data, labels, test_labels = get_train_test_data(mnist)
    knn_clf = Pipeline([("knn", KNeighborsClassifier(n_neighbors=4, weights="distance"))])
    knn_clf.fit(data, labels)
    test_predictions = knn_clf.predict(test_data)
    return accuracy_score(test_predictions, test_labels)


def q2():
    mnist = get_mnist_data()
    data, test_data, labels, test_labels = get_train_test_data(mnist)
    data, labels = augment_data(data, labels)
    knn_clf = Pipeline([("knn", KNeighborsClassifier(n_neighbors=4, weights="distance"))])
    knn_clf.fit(data, labels)
    test_predictions = knn_clf.predict(test_data)
    return accuracy_score(test_predictions, test_labels)


def augment_data(data, labels):
    data = [im.reshape((28, 28)) for im in data]
    data = (
        [
            shift(im, shift_value, mode="constant", cval=0)
            for shift_value in [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
            for im in data
        ]
    )
    data = numpy.array([im.reshape((28 * 28, )) for im in data])
    labels = numpy.tile(labels, 5)
    return data, labels


def q3():
    # Initial thoughts about the data
    # Pclass is social class rather than cabin class, this could be important
    # Name probably isn't important, but I could 1 hot encode surname, to see if there's correlation between families
    # Actually this is covered by SibSp and Parch, so probably drop name
    # Sex is probably important
    # Age is probably important
    # SibSp is number of siblings and spouses, having this means I can probably drop name
    # Parch is the number of parents and children
    # Ticket is probably irrelevant. I don't want to go too deep on it so drop.
    # Cabin is probably very relevant. I could separate the deck from the number as 2 features, but it's very
    # incomplete so ignore for now.
    # Embarked is C = Cherbourg, Q = Queenstown, S = Southampton. Probably not relevant unless it determines something
    # else about the passengers. One hot encode.

    data, test_data, labels = load_titanic_data()
    preprocessing = get_preprocessing_pipeline()
    rf_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("rf", RandomForestClassifier(random_state=0))
        ]
    )
    param_grid = [
        {
            'rf__criterion': ["gini", "entropy", "log_loss"],
            "rf__max_depth": [None, 10, 100, 1000],
            "rf__min_weight_fraction_leaf": [0.0, 0.1, 0.2]
         },
    ]
    search = RandomizedSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        random_state=0,
        n_iter=10
    )
    search.fit(data, labels)
    return pandas.DataFrame(search.cv_results_)


def get_preprocessing_pipeline():
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return ColumnTransformer(
        [
            ("num", num_pipeline, ["Pclass", "Age", "SibSp", "Parch", "Fare"]),
            ("cat", cat_pipeline, ["Sex", "Embarked"]),
        ],
        remainder="drop"
    )


def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    with tarfile.open(tarball_path) as tarball:
        tarball.extractall(path="datasets")
    train = pandas.read_csv(Path("datasets/titanic/train.csv"))
    y_train = train["Survived"]
    x_train = train.drop(columns="Survived")
    x_test = pandas.read_csv(Path("datasets/titanic/test.csv"))
    return x_train, x_test, y_train
