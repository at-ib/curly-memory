# 1 & 2. The fundamental idea behind support vector machines is fitting the widest possible 'street' between different
# classes. The descision boundary is in the midde of the street. The descision boundary only depends on the
# locations of the 'support vector' ie those that on the edge of or within the street.

# 3. SVMs are sensitive to feature scales. It is only possible to get the widest street with scaled vectors

# 4. Yes it can output a probability, but to do so requires multiple folds of cross validation during training.

# 5. LinearSVC doesn't provide probabilities, so use SVC if you want probabilities. LinearSVC and SGDClassifier have
# lower complexities than SVC, so use for large training sets. Only SVC supports hte kernel trick, and only
# SGDClassifier has out of core support.

# 6. If the model is underfitting increase gamma. Same for C.

# 7. A model is epsilon-insensitive if when adding more training instances they are all within the margin.

# 8. Kernel trick allows is equivalent to adding extra nonlinear features, but without actually adding them. This is
# useful when the data isn't linearly seperable.
import numpy
import pandas
from scipy.stats import loguniform
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR


def q9():
    data, test_data, labels, test_labels = get_q9_data()
    lsvc_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lsvc", LinearSVC(C=1, random_state=42, dual="auto"))
        ]
    )
    lsvc_clf.fit(data, labels)
    print(f"lsvc_clf coef_: {lsvc_clf.steps[1][1].coef_}")
    print(f"lsvc_clf intercept_: {lsvc_clf.steps[1][1].intercept_}")
    sgdc_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lsvc", SGDClassifier(random_state=42, alpha=0.1))
        ]
    )
    sgdc_clf.fit(data, labels)
    print(f"sgdc_clf coef_: {sgdc_clf.steps[1][1].coef_}")
    print(f"sgdc_clf intercept_: {sgdc_clf.steps[1][1].intercept_}")


def get_q9_data():
    iris = load_iris(as_frame=True)
    data = iris.data
    data = data.loc[:, data.columns.str.startswith("petal")]
    labels = iris.target
    data = data.join(labels)
    data = data[data["target"] < 2]  # If we drop category 2 the data is linearly seperable
    return train_test_split(
        data.loc[:, data.columns.str.startswith("petal")],
        data["target"],
        random_state=0,
    )


def q10():
    wine = load_wine(as_frame=True)
    data, test_data, labels, test_labels = train_test_split(wine.data, wine.target, random_state=0, stratify=wine.target)
    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=0, probability=True))
        ]
    )
    param_grid = [
        {
            'svc__kernel': ["linear", "rbf", "poly", "sigmoid"],
            "svc__degree": [2, 3, 4, 5],
            "svc__gamma": loguniform(1e-6, 1e1),
            "svc__C": loguniform(1e-6, 1e1),
         },
    ]
    search = RandomizedSearchCV(
        svm_clf,
        param_grid,
        cv=3,
        scoring='accuracy',
        random_state=0,
        n_iter=1000
    )
    search.fit(data, labels)
    print(f"best score: {search.best_score_}")
    print(f"accuracy on test set: {search.score(test_data, test_labels)}")


def q11():
    housing = fetch_california_housing(as_frame=True)
    target_bins = pandas.cut(
        housing.target,
        bins=[0., 1.5, 3.0, 4.5, 6., numpy.inf],
        labels=[1, 2, 3, 4, 5]
    )
    data, test_data, labels, test_labels = train_test_split(
        housing.data,
        housing.target,
        random_state=0,
        stratify=target_bins
    )
    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR())
        ]
    )
    param_grid = [
        {
            "svr__gamma": loguniform(0.001, 0.1),
            "svr__C": loguniform(1, 10),
         },
    ]
    search = RandomizedSearchCV(
        svm_clf,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=0,
        n_iter=200
    )
    search.fit(data[:2000], labels[:2000])
    print(f"best score: {search.best_score_}")
    print(f"accuracy on test set: {search.score(test_data, test_labels)}")

