from pathlib import Path

import numpy
import pandas
import pandas as pd
import tarfile
import urllib.request

import sklearn.model_selection
from sklearn.impute import SimpleImputer


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


housing = load_housing_data()


def get_data_splits(housing_data):
    return sklearn.model_selection.train_test_split(housing_data, train_size=5000, random_state=0)


def prepare_data(data):
    data = impute_missing_values(data)
    data = encode_categoricals(data)
    return data


def impute_missing_values(data):
    imputer = SimpleImputer(strategy="median")
    data_num = data.select_dtypes(include=numpy.number)
    data_nonnum = data.select_dtypes(exclude=numpy.number)
    data_num = pandas.DataFrame(imputer.fit_transform(data_num), columns=data_num.columns, index=data_num.index)
    return pandas.concat([data_num, data_nonnum], axis=1)


def encode_categoricals(data):
    return data
