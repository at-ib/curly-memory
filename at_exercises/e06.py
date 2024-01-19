# 1. 20? Or does it depend on the labels?

# 2. Always lower. WRONG, generally lower

# 3. Yes

# 4. Scaling the input features makes no difference to a descision tree.

# 5. 1,000,000 log(1,000,000) = k * 3600. k = 1,000,000 log(1,000,000)/3600
# 10,000,000 log(10,000,000) = kt. t = 10,000,000 log(10,000,000)/k = 10 * 3600 * log(10,000,000)/log(1,000,0000)
# = 3.3 * 10 * 3600, something like 33 hours. WRONG 11.7 hours

# 6. 2 hours


import numpy
import scipy
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier


def q7():
    moons = make_moons(n_samples=1000, noise=0.4)
    data, test_data, labels, test_labels = train_test_split(moons[0], moons[1])
    dt_clf = DecisionTreeClassifier(random_state=0)
    param_grid = [
        {
            'min_samples_split': numpy.arange(2, 11),
            'max_leaf_nodes': numpy.arange(2, 11),
            'min_samples_leaf': numpy.arange(2, 11),
         },
    ]
    search = GridSearchCV(dt_clf, param_grid, cv=3)
    search.fit(data, labels)
    print(f"best score: {search.best_score_}")
    print(f"best params: {search.best_params_}")
    df_clf_best = DecisionTreeClassifier(random_state=0, max_leaf_nodes=4, min_samples_leaf=2, min_samples_split=2)
    df_clf_best.fit(data, labels)
    print(f"accuracy on test set: {search.score(test_data, test_labels)}")


def q8():
    n_trees = 1000
    n_instances = 100
    moons = make_moons(n_samples=1000, noise=0.4)
    data, test_data, labels, test_labels = train_test_split(moons[0], moons[1])
    splitter = ShuffleSplit(n_splits=n_trees, train_size=n_instances)
    split_indices = list(splitter.split(data, labels))
    split_data = [data[ind[0]] for ind in split_indices]
    split_labels = [labels[ind[0]] for ind in split_indices]
    forest = [
        DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=2, min_samples_split=2)
        for _ in range(n_trees)
    ]
    for i in range(n_trees):
        forest[i].fit(split_data[i], split_labels[i])
    for i in range(10):
        print(f"accuracy of {i}th tree on test set: {forest[i].score(test_data, test_labels)}")
    predictions = [[forest[i].predict(test_data[j].reshape(1, -1)) for i in range(n_trees)] for j in range(len(test_data))]
    predictions = [scipy.stats.mode(pred)[0] for pred in predictions]
    print(f"forest accuracy: {accuracy_score(predictions, test_labels)}")



