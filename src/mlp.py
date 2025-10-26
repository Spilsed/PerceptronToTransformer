import numpy as np
import numpy.typing as npt
from typing import Tuple

from activations import ReLU, Softmax
from losses import MSE


class MLP():
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.hidden_weights = np.random.randn(
            self.input_size, self.hidden_size
        )
        self.hidden_bias = np.random.randn(1, self.hidden_size)

        self.output_weights = np.random.randn(
            self.hidden_size, self.output_size
        )
        self.output_bias = np.random.randn(1, self.output_size)

        self.relu = ReLU()
        self.softmax = Softmax()
        self.mse = MSE()

    def feedforward_layers(self, x: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        z0 = x @ self.hidden_weights + self.hidden_bias
        a0 = self.relu(z0)
        z1 = a0 @ self.output_weights + self.output_bias

        return z0, a0, z1

    def feedforward(self, x: npt.NDArray) -> npt.NDArray:
        x = x @ self.hidden_weights + self.hidden_bias
        x = self.relu(x)
        x = x @ self.output_weights + self.output_bias

        return self.softmax(x)

    def backprop(self, x: npt.NDArray, y: npt.NDArray) -> None:
        if len(x.shape) < 2:
            x = np.array([x])
        batches = x.shape[0]

        z0, a0, z1 = self.feedforward_layers(x)

        dz1 = self.mse.grad(z1, y)
        dw1 = a0.T @ dz1 / batches
        db1 = np.sum(dz1, axis=0, keepdims=True)

        da1 = dz1 @ self.output_weights.T
        dz0 = da1 * self.relu.grad(z0)
        dw0 = x.T @ dz0 / batches
        db0 = np.sum(dz0, axis=0, keepdims=True)

        self.output_weights -= dw1 * self.learning_rate
        self.output_bias -= db1 * self.learning_rate

        self.hidden_weights -= dw0 * self.learning_rate
        self.hidden_bias -= db0 * self.learning_rate

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        return self.feedforward(x)


if __name__ == "__main__":
    from datasets import generate_blobs
    import matplotlib.pyplot as plt

    x_train, y_train, x_test, y_test = generate_blobs(5000, 4, 4, 5)

    mlp = MLP(4, 16, 4, 1e-2)
    mse = MSE()

    losses = []
    for _ in range(50):
        for x, y in zip(x_train, y_train):
            mlp.backprop(x, y)
        losses.append(mse(mlp(x_test), y_test))

    pred = mlp(x_test)

    accuracy = np.mean(np.argmax(pred, axis=-1) == np.argmax(y_test, axis=-1))
    print(accuracy, losses[-1])

    plt.plot(losses)
    plt.show()
