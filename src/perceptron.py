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
    from losses import MSE

    x_train, y_train, x_test, y_test = generate_blobs(one_hot=False, seed=9)

    perceptron = Perceptron(2)

    w1_weights = []
    w2_weights = []
    losses = []

    loss_function = MSE()
    for _ in range(50):
        for inputs, targets in zip(x_train, y_train):
            perceptron.backprop(inputs, targets)
        w1_weights.append(perceptron.weights[1])
        w2_weights.append(perceptron.weights[2])

        pred = perceptron(x_test)
        losses.append(loss_function(pred, y_test))

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

    final_weights = perceptron.weights.copy()

    GRID_SIZE = 101
    DELTA_RANGE_1 = 0.8
    DELTA_RANGE_2 = 0.5
    w1_values = np.linspace(
        perceptron.weights[0] - DELTA_RANGE_1, perceptron.weights[0] + DELTA_RANGE_1, GRID_SIZE)
    w2_values = np.linspace(
        perceptron.weights[1] - DELTA_RANGE_2, perceptron.weights[1] + DELTA_RANGE_2, GRID_SIZE)

    w1_grid, w2_grid = np.meshgrid(w1_values, w2_values)
    weight_plane = np.stack((w1_grid, w2_grid), axis=2)

    loss_plane = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            current_w1 = w1_grid[i, j]
            current_w2 = w2_grid[i, j]

            perceptron.weights[1] = final_weights[1] + current_w1
            perceptron.weights[2] = final_weights[2] + current_w2

            pred = perceptron(x_test)
            loss = loss_function(pred, y_test)

            loss_plane[i, j] = loss

    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection='3d')

    ax.plot_surface(w1_grid, w2_grid, loss_plane, cmap='viridis', alpha=0.6)

    ax.plot3D(
        w1_weights, w2_weights, losses,
        color='red',
        linewidth=3,
        marker='o',
        markersize=5,
        label='Training Trajectory'
    )

    ax.scatter(w1_weights[-1], w2_weights[-1], losses[-1],
               color='magenta', marker='*', s=200, label='End Point')

    ax.set_xlabel('$\\Delta$Weight 1 ($w_1$)')
    ax.set_ylabel('$\\Delta$Weight 2 ($w_2$)')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Surface')

    plt.show()
