

# 1. Clustering is grouping similar instances together without having any information about what the instances are. A
# few clustering algorithms: k-means, DBSCAN, Gaussian mixture models.

# 2. The main applications of clustering algorithms are data analysis, customer segmentation, recommender systems,
# search engines, image segmentation, semi-supervised learning, dimensionality reduction, and more.

# 3. Technique 1: Look for the elbow in the intertia-k chart. Technique 2: silhouette diagram

# 4. When you have a partially labelled dataset, you can do semi-supervised learning by propogating the labels from
# labelled training instances to unlabelled training instances. This can be done using clustering.

# 5. Scalable: k-means and BIRCH. Looking for regions of high density: DBSCAM and mean-shift.

# 6. Active learning is useful in labelling people in photos. The algorithm clusters similar faces together, but when it
# is uncertain it asks a human to decide whether the faces are the same person or not.

# 7. Anomaly detection is detecting instances which are different from the standard instances in the training set.
# Novelty detection is similar, but it assumes that everything in the dataset is clean and good and it looks for
# totally new examples.

# 8. A Gaussian mixture model assumes that the data comes from a mixture of Gaussian provability distributions. The
# model tries to find the parameters for these distributions. You can use it for clustering and anomaly detection.

# 9. To find the right number of clusters minimise a theoretical information criterion, like the Akaike information
# criterion or the Bayesian information criterion.
import pickle

import numpy
from matplotlib import pyplot
from numpy import argmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


def q10():
    olivetti = get_olivetti()
    data, test_data, labels, test_labels = train_test_split(
        olivetti.data, olivetti.target, random_state=0, stratify=olivetti.target, test_size=2 * 40
    )
    data, validation_data, labels, validation_labels = train_test_split(
        data, labels, random_state=0, stratify=labels, test_size=2 * 40
    )
    # k = get_best_k(data)
    k = 80
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(data)
    show_faces_for_cluster(data, labels, kmeans.labels_, 0)
    show_faces_for_cluster(data, labels, kmeans.labels_, 9)
    show_faces_for_cluster(data, labels, kmeans.labels_, 19)
    show_faces_for_cluster(data, labels, kmeans.labels_, 29)
    show_faces_for_cluster(data, labels, kmeans.labels_, 39)


def q11():
    olivetti = get_olivetti()
    data, test_data, labels, test_labels = train_test_split(
        olivetti.data, olivetti.target, random_state=0, stratify=olivetti.target, test_size=2 * 40
    )
    data, validation_data, labels, validation_labels = train_test_split(
        data, labels, random_state=0, stratify=labels, test_size=2 * 40
    )
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(data, labels)
    print(f"rf_clf score: {rf_clf.score(validation_data, validation_labels)}")
    kmeans = KMeans(n_clusters=80, random_state=0, n_init="auto")
    kmeans.fit(data)
    data_reduced = kmeans.transform(data)
    validation_data_reduced = kmeans.transform(validation_data)
    rf_clf_reduced = RandomForestClassifier(random_state=1)
    rf_clf_reduced.fit(data_reduced, labels)
    print(f"rf_clf_reduced score: {rf_clf_reduced.score(validation_data_reduced, validation_labels)}")
    pipeline = Pipeline(
        [
            ("kmeans", KMeans(random_state=0, n_init="auto")),
            ("rf_clf", RandomForestClassifier(random_state=0))
        ]
    )
    param_grid = [
        {
            "kmeans__n_clusters": range(10, 100, 20)
        }
    ]
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
    search.fit(data, labels)
    print(
        (
            f"Using dim reduction, grid search found that the best k is {search.best_params_['kmeans__n_clusters']} "
            f"which gives a score of {search.best_score_}"
        )
    )
    data_extended = numpy.c_[data, data_reduced]
    validation_data_extended = numpy.c_[validation_data, validation_data_reduced]
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(data_extended, labels)
    print(
        "Extending the data with the kmeans features gives a score of "
        f"{rf_clf.score(validation_data_extended, validation_labels)}"
    )


def q12():
    olivetti = get_olivetti()
    data = olivetti.data
    pca = PCA(n_components=0.99, random_state=0)
    data_reduced = pca.fit_transform(data)
    gm = GaussianMixture(random_state=0, n_components=40)
    gm.fit(data_reduced)
    sam_pca = gm.sample()
    sam = pca.inverse_transform(sam_pca[0])
    show_face(sam)
    modified_face = data[:2] / 2
    modified_pca = pca.transform(modified_face)
    print(f"Scores for original faces: {gm.score_samples(data_reduced)}")
    print(f"Score for modified faces: {gm.score_samples(modified_pca)}")


def q13():
    olivetti = get_olivetti()
    data = olivetti.data
    pca = PCA(n_components=0.99, random_state=0)
    data_reduced = pca.fit_transform(data)
    data_reconstructed = pca.inverse_transform(data_reduced)
    print(f"Reconstruction error: {mean_squared_error(data, data_reconstructed)}")
    modified_face = data[:2] / 2
    modified_reduced = pca.transform(modified_face)
    modified_reconstructed = pca.inverse_transform(modified_reduced)
    print(f"Modified reconstruction error: {mean_squared_error(modified_face, modified_reconstructed)}")
    show_face(modified_reconstructed[0])


def get_best_k(data):
    sil_score = {k: get_silhouette_score(data, k) for k in range(20, 160, 20)}
    print(sil_score)
    k = max(sil_score, key=sil_score.get)
    print(f"The silhouette score suggests that the optimum value of k is {k}")
    return k


def get_olivetti():
    with open("datasets/olivetti.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def show_face(face):
    face = face.reshape(64, 64)
    pyplot.imshow(face, cmap="binary_r")


def get_silhouette_score(data, k):
    print(f"Calculating silhouette score for k={k}")
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(data)
    return silhouette_score(data, kmeans.labels_)


def show_faces_for_cluster(data, labels, cluster_labels, cluster):
    faces = data[cluster_labels == cluster]
    actual_labels = labels[cluster_labels == cluster]
    plot_faces(faces, actual_labels)


def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    pyplot.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        pyplot.subplot(n_rows, n_cols, index + 1)
        pyplot.imshow(face, cmap="gray")
        pyplot.axis("off")
        pyplot.title(label)
