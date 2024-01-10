from pathlib import Path

import numpy as np
import pandas as pd
import tarfile
import urllib.request

import sklearn.model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# TRAIN_SIZE = 5000
TRAIN_SIZE = 0.2

CAT_COL = "ocean_proximity"


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def process_data_pre_split(data):
    return data.assign(income_categories=get_income_categories(data["median_income"]))


def get_data_splits(data, stratify):
    return sklearn.model_selection.train_test_split(data, train_size=TRAIN_SIZE, stratify=stratify, random_state=0)


def get_income_categories(income_data):
    return pd.cut(
        income_data,
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def get_preprocessing_pipeline():

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler())
    return ColumnTransformer([
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                   "households", "median_income"]),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_num_pipeline)


if __name__ == "__main__":
    data = load_housing_data()
    data = process_data_pre_split(data)
    data, test_data = get_data_splits(data.drop(columns="income_categories"), stratify=data["income_categories"])
    labels = data["median_house_value"]
    data = data.drop(columns="median_house_value")
    preprocessing = get_preprocessing_pipeline()
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(data, labels)
    lin_predictions = lin_reg.predict(data)
    lin_rmse = mean_squared_error(labels, lin_predictions, squared=False)
    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(data, labels)
    tree_predictions = tree_reg.predict(data)
    lin_rmses = cross_val_score(lin_reg, data, labels, scoring="neg_root_mean_squared_error")
    tree_rmses = cross_val_score(tree_reg, data, labels, scoring="neg_root_mean_squared_error")
    print(f"lin_rmses: {lin_rmses}")
    print(f"tree_rmses: {tree_rmses}")

