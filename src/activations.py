import numpy as np
import numpy.typing as npt

from typing import TypeAlias


FloatArray: TypeAlias = npt.NDArray[np.float32 | np.float64]


def heavy_side_step(x: npt.NDArray) -> npt.NDArray:
    return np.clip(x, 0.0, 1.0)


class Activation():
    def __call__(self, x: FloatArray) -> FloatArray:
        return np.array([])

    def grad(self, x: FloatArray) -> FloatArray:
        return np.array([])


class ReLU(Activation):
    def __call__(self, x: FloatArray) -> FloatArray:
        return np.clip(x, 0.0, None)

    def grad(self, x: FloatArray) -> FloatArray:
        return np.where(x < 0, 0, 1)


class Softmax(Activation):
    def __call__(self, x: FloatArray) -> FloatArray:
        ex = np.e ** (x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def grad(self, x: FloatArray) -> FloatArray:
        p = self(x)
        return p * (1 - p)
