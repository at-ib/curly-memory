# 1. Gradient descent

# 2. Gradient descent suffers with mismatched scales. Standardizing features helps with this.

# 3. No, because the cost function is convex

# 4. No. Stochastic gradient descent probably won't reach the optimum, it will just get close

# 5. The learning rate is too high

# 6. No, because with mini batch you sometimes go uphill in a single step

# 7. The algorithm that will get to the vicinity of the optimum first is stochastic? Batch gradient descent will
# actually converge. Others can be made to converge by reducing the learning rate over time.

# 8. If there is a large gap between the learning curves for the training and validation data then the model is
# overfitting. This can be improved with more data, more regularization or a simpler model.

# 9. If training and validation error are both high the model is underfitting (ie high bias). In which case we want less
# regularization. For ridge regression, this means we should reduce alpha.

# 10a. Use ridge regression to constrain the weights of the model and therefore reduce overfitting.

# 10b. Use lasso regression when you suspect only a few features are important because it tends to set most of the
# weights to zero

# 10c. Elastic net is a compromise between ridge regression and lasso regression. In general, elastic net is preferred
# over lasso because lasso may behave erratically when the number of features is greater than the number of training
# instances or when several features are strongly correlated.

# 11. 2 logistic regression classifiers. Softmax classifiers are not multioutput.
import numpy
from numpy import dot, hstack, exp, arange, argmax, random
from numpy.random import rand, randn
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def q12():
    random.seed(0)
    iris = load_iris(as_frame=True)
    data, test_data, labels, test_labels = train_test_split(
        iris.data,
        iris.target,
        random_state=0,
        stratify=iris.target
    )
    theta_initial = randn(5, 3)
    learning_rate = 0.1
    tol = 0.00001
    n_epochs = 5000
    return gradient_descent(theta_initial, data, labels, learning_rate, tol, n_epochs)


def predict(theta, x):
    # x_i is the ith training instance
    return argmax([softmax(k, theta, x) for k in arange(theta.shape[1])])


def softmax(k, theta, x):
    num = exp(score(k, theta, x))
    den = numpy.sum([exp(score(j, theta, x)) for j in arange(theta.shape[1])])
    return num / den


def score(k, theta, x):
    return dot(theta[:, k], hstack((1, x)))


def cost(theta, data, labels):
    num = numpy.sum([numpy.log(softmax(labels.iloc[i], theta, data.iloc[i, :])) for i in numpy.arange(len(data))])
    den = -len(data)
    return num / den


def gradient(theta, data, labels):
    k_range = numpy.arange(theta.shape[1])
    return numpy.array([gradient_k(k, theta, data, labels) for k in k_range]).T


def gradient_k(k, theta, data, labels):
    m_range = numpy.arange(len(data))
    num = numpy.sum(
        [(softmax(k, theta, data.iloc[i, :]) - indicator(k, i, labels)) * hstack((1, data.iloc[i, :])) for i in m_range],
        axis=0
    )
    den = len(data)
    return num / den


def indicator(k, i, labels):
    return int(labels.iloc[i] == k)


def theta_next(theta, data, labels, learning_rate):
    return theta - learning_rate * gradient(theta, data, labels)


def gradient_descent(theta_initial, data, labels, learning_rate, tol, n_epochs):
    prev_cost = cost(theta_initial, data, labels)
    new_cost = prev_cost - 2 * tol
    theta = theta_initial
    i = 0
    while (new_cost <= prev_cost) and i < n_epochs:
        prev_cost = new_cost.copy()
        theta = theta_next(theta, data, labels, learning_rate)
        new_cost = cost(theta, data, labels)
        if i % 1000 == 0:
            print(f"step_{i}")
            print(f"cost: {new_cost}")
        i += 1
    return theta
