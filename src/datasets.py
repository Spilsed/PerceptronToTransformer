import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

from typing import Tuple


def generate_blobs(n_samples: int = 1000, n_features: int = 2, centers: int = 2, cluster_std: int = 3, test_size: float = 0.2, one_hot: bool = True) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    x, y = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=centers,
                      cluster_std=cluster_std)[:2]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=True)

    # Standardize the data to have a zero mean and unit variance
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if one_hot:
        y_train = one_hot_encode(y_train)
        y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test


def one_hot_encode(y: npt.NDArray) -> npt.NDArray:
    b = np.zeros((y.size, y.max() + 1))
    b[np.arange(y.size), y] = 1

    return b
