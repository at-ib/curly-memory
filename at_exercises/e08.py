# 1. The main motivations for dimensionality reduction are speeding up training and data viz. The main drawbacks are
# that it causes information loss, which may result in a worse model and makes pipelines more complex.

# 2. The curse of dimensionality is that in high dimensions there's a lot of space and datasets tend to be
# very sparse. This means training instances and new data tend to be far away. All this leads to models which are likely
# to overfit.

# 3. No, once dimensionality has been reduced that information is lost.

# 4. No, PCA relies on projection, so it wouldn't work well for something like the swiss roll which requires manifold
# learning

# 5. Depends on the data. It could be 1 dimension or you could need the first 950 dimensions

# 6. Use incremental PCA when the full dataset won't fit in memory. Going from regular PCA to randomized PCA to
# randomised projection greatly reduces the computational complexity, but for slightly worse outcomes. So us the more
# randomized ones when the datasets are larger.

# 7. Apply the inverse transform and see what the reconstruction error is.

# 8. Yes?
from time import perf_counter

import numpy
import pandas
from matplotlib import pyplot
from matplotlib.pyplot import show
from sklearn import clone
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
from sklearn.model_selection import train_test_split


def q9():
    mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
    data, test_data, labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=0)
    print("RandomForestClassifier:")
    compare_full_data_with_pca(data, labels, test_data, test_labels, RandomForestClassifier(random_state=0))
    print("SGDClassifier:")
    compare_full_data_with_pca(data, labels, test_data, test_labels, SGDClassifier(random_state=0))


def compare_full_data_with_pca(data, labels, test_data, test_labels, classifier):
    train_start = perf_counter()
    classifier.fit(data, labels)
    train_stop = perf_counter()
    print(
        f"classifier without PCA took {train_stop - train_start}s to train and has an accuracy "
        f"of {classifier.score(test_data, test_labels)}"
    )
    pca = PCA(n_components=0.95)
    data_reduced = pca.fit_transform(data)
    reduced_classifier = clone(classifier)
    train_start = perf_counter()
    reduced_classifier.fit(data_reduced, labels)
    train_stop = perf_counter()
    test_data_reduced = pca.transform(test_data)
    print(
        f"reduced_classifier took {train_stop - train_start}s to train and has an accuracy "
        f"of {reduced_classifier.score(test_data_reduced, test_labels)}"
    )


def q10():
    mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
    data, test_data, labels, test_labels = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=0)
    data = data[:5000, ]
    labels = labels[:5000]
    dim_reduce_and_plot(data, labels, TSNE(random_state=0))
    dim_reduce_and_plot(data, labels, PCA(random_state=0, n_components=2))
    dim_reduce_and_plot(data, labels, LocallyLinearEmbedding(random_state=0, n_components=2))
    dim_reduce_and_plot(data, labels, MDS(random_state=0, n_components=2))


def dim_reduce_and_plot(data, labels, algorithm):
    data = algorithm.fit_transform(data)
    data = pandas.DataFrame(data, columns=["feat0", "feat1"])
    labels_frame = pandas.DataFrame(labels, columns=["labels"]).astype("float")
    data = data.join(labels_frame)
    data.plot.scatter(x="feat0", y="feat1", c="labels")

