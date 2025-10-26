import numpy as np
import numpy.typing as npt


class Perceptron():
    def __init__(self, input_size: int, learning_rate: float = 1e-2) -> None:
        self.input_size = input_size
        self.learning_rate = learning_rate

        self.weights = np.random.rand(input_size + 1)

    def heavy_side_step(self, x: float) -> float:
        if (x < 0):
            return 0.0
        else:
            return 1.0

    def backprop(self, x: npt.NDArray, y: float) -> None:
        z = self(x)

        self.weights[1:] -= self.learning_rate * (z - y) * x
        self.weights[0] -= self.learning_rate * (z - y)[0]

    def train(self, x: npt.NDArray, y: npt.NDArray, epochs: int) -> None:
        for _ in range(epochs):
            for inputs, targets in zip(x, y):
                self.backprop(inputs, targets)

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        if len(x.shape) < 2:
            x = np.array([x])

        Z = x @ self.weights[1:].T + self.weights[0]

        outputs = []
        for i in range(len(Z)):
            outputs.append(self.heavy_side_step(Z[i]))

        return np.array(outputs)


if __name__ == "__main__":
    from datasets import generate_blobs
    import matplotlib.pyplot as plt

    x_train, y_train, x_test, y_test = generate_blobs(one_hot=False)

    perceptron = Perceptron(2)
    perceptron.train(x_train, y_train, epochs=50)

    pred = perceptron(x_test)

    accuracy = np.mean(pred == y_test)

    plt.title(f'Accuracy: {accuracy}')
    # Plot the different points
    plt.scatter(x_test[:, 0], x_test[:, 1], c=pred + 2)
    # Show the decision boundary
    plt.axline((0, -(perceptron.weights[0]/perceptron.weights[2])), slope=-
               perceptron.weights[1]/perceptron.weights[2])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
