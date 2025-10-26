import numpy as np
import numpy.typing as npt


class Loss():
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        return 0


class MSE(Loss):
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        return float(np.sum(np.pow(x - y, 2)))

    def grad(self, x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        return (x - y) * 2 / y.shape[0]
