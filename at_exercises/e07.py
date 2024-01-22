# 1. Yes, you should be able to use the different models to vote on the answers and increase your precision

# 2. Hard voting is when each classifier says what it thinks the answer is and the answer with the most votes is
# selected. In soft voting each classifier outputs a probability for each outcome and the highest average probability
# is taken to be the answer.

# 3. Bagging, pasting, random forest can all be sped up with multiple servers. Boosting cannot be. Stacking can be for
# the parallel layers but not the blender.

# 4. Out of bag evaluation can be used for model validation without having to have a separate validation set, which
# means more data is available for training.

# 5. Random thresholds are used in the trees, rather than the normal approach of computing the threshold. This trades
# more bias for lower variance. It also makes the model much quicker to train.

# 6. If your AdaBoost ensemble is underfitting the training set, you can try incrasing the number of estimators or less
# strongly regularizing the base estimator.

# 7. If gradient boosting algorithm overfits reduce the learning_rate hyperparameter
import numpy
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def q8():
    mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
    data, test_data, labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=0)
    data, validation_data, labels, validation_labels = train_test_split(data, labels, test_size=10000, random_state=0)
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(data, labels)
    et_clf = ExtraTreesClassifier(random_state=0)
    et_clf.fit(data, labels)
    # sv_clf = SVC(random_state=0, probability=True)
    sv_clf = SVC(random_state=0, probability=False)
    sv_clf.fit(data, labels)
    print(f"rf_clf accuracy on validation set: {rf_clf.score(validation_data, validation_labels)}")
    print(f"et_clf accuracy on validation set: {et_clf.score(validation_data, validation_labels)}")
    print(f"sv_clf accuracy on validation set: {sv_clf.score(validation_data, validation_labels)}")
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", rf_clf),
            ("et", et_clf),
            ("sv", sv_clf)
        ]
    )
    # voting_clf.voting = "soft"
    voting_clf.fit(data, labels)
    print(f"voting_clf accuracy on validation set: {voting_clf.score(validation_data, validation_labels)}")


def q9():
    mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
    data, test_data, labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=0)
    data, validation_data, labels, validation_labels = train_test_split(data, labels, test_size=10000, random_state=0)
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(data, labels)
    et_clf = ExtraTreesClassifier(random_state=0)
    et_clf.fit(data, labels)
    sv_clf = SVC(random_state=0)
    sv_clf.fit(data, labels)
    layer_0_predictions = (
        rf_clf.predict(validation_data),
        et_clf.predict(validation_data),
        sv_clf.predict(validation_data),
    )
    blender_data = numpy.concatenate([pred.reshape(-1, 1) for pred in layer_0_predictions], axis=1)
    rf_blender = RandomForestClassifier(random_state=1)
    rf_blender.fit(blender_data, validation_labels)
    blender_test_predictions = get_blender_predictions(test_data, [rf_clf, et_clf, sv_clf], rf_blender)
    print(f"Blender accuracy: {accuracy_score(blender_test_predictions, test_labels)}")


def get_blender_predictions(data, layer_0, blender):
    layer_0_predictions = (clf.predict(data) for clf in layer_0)
    layer_0_predictions = numpy.concatenate([pred.reshape(-1, 1) for pred in layer_0_predictions], axis=1)
    return blender.predict(layer_0_predictions)


def q9_stacking_classifier():
    mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
    data, test_data, labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=0)
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=0)),
            ('et', ExtraTreesClassifier(random_state=0)),
            ('svc', SVC(random_state=0))
        ],
        final_estimator=RandomForestClassifier(random_state=1),
    )
    stacking_clf.fit(data, labels)
    print(f"StackingClassifier accuracy: {stacking_clf.score(test_data, test_labels)}")
