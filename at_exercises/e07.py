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
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
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

