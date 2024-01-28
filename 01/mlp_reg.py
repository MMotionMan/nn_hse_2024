import math
import numpy as np


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, arr):
        np.array([self.__call__(el) for el in arr])

class Relu:
    def __call__(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype("float")


class BCELoss:
    def forward(self, predict, y):
        p = sigmoid_for_el(predict)
        return -1 * (y * np.log(p + 1e-5) + (1 - y) * np.log(1 - p + 1e-5))

    def backward(self, predict, y):
        p = sigmoid_for_el(predict)
        return np.expand_dims(p - y, -1)


class LinearLayer:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.grad_w = None
        self.grad_b = None
        self.activation = Relu()
        self.last_input = None
        self.last_out = None

    def calc(self, x):
        self.last_input = x
        out = x @ self.w + self.b
        if self.activation:
            self.last_out = out
            out = self.activation(out)

        return out

    def backward(self, grad):
        if self.activation:
            grad = np.multiply(grad, self.activation.backward(self.last_out))

        self.grad_w = np.transpose(self.last_input) @ grad
        self.grad_b = np.sum(grad, axis=0)
        return grad @ np.transpose(self.w)

    def weights_update(self, alpha=1e-2):
        self.w = self.w.astype("float64")
        self.w -= alpha * self.grad_w
        self.b = self.b.astype("float")
        self.b -= alpha * self.grad_b


class MLPRegressor:
    def __init__(self, x_input, layers_dims, loss_function=None):
        """
        :param x_input:
        :param layers_count:
        :param loss_function:
        """

        self.layers_count = len(layers_dims)
        self.loss_function = loss_function
        self.input_size = len(x_input[0])
        self.layers_dims = layers_dims
        self.weights, self.bias = self.initialize_parameters(self.layers_dims)
        self.x_input = x_input
        self.hidden_layer = np.array(
            [LinearLayer(self.weights[i], self.bias[i]) for i in range(len(self.weights))]
        )

    def initialize_parameters(self, layer_dims):
        length = len(layer_dims)
        w = [np.random.normal(loc=0, scale=0.2, size=(layer_dims[i - 1], layer_dims[i])) for i in range(1, length)]
        b = [np.zeros((1, layer_dims[i])) for i in range(1, length)]
        return w, b

    def forward_pass(self, x):
        layer_output = x
        for layer in self.hidden_layer:
            layer_output = layer.calc(layer_output)
        return layer_output.ravel()

    def backward_pass(self, p, y, loss_function):
        loss = loss_function()
        loss_value = loss.forward(p, y)
        grad = loss.backward(p, y)
        for layer in reversed(self.hidden_layer):
            grad = layer.backward(grad)
            layer.weights_update()

        return loss_value


def sigmoid(x):
    return 1 / (1 + math.exp(-x + 1e-5))


def sigmoid_for_el(arr):
    return np.array([sigmoid(el) for el in arr])

N = 200
N2 = 25

n_pos = int(N // 2 + np.random.randint(-N2, N2))
n_neg = int(N // 2 + np.random.randint(-N2, N2))

pos_x = 1
pos_y = 1

neg_x = -1
neg_y = -1

pos_pairs = np.array([np.array(
    [pos_x + np.random.normal(scale=0.2), pos_y + np.random.normal(scale=0.2)])
    for i in range(0, n_pos)])

pos_answers = np.array([1] * n_pos)

neg_pairs = np.array([np.array(
    [neg_x + np.random.normal(scale=0.2), neg_y + np.random.normal(scale=0.2)]) for i in range(0, n_neg)]
)
neg_answers = np.array([0] * n_neg)

x = np.vstack([pos_pairs, neg_pairs])
y = np.hstack([pos_answers, neg_answers])

network = MLPRegressor(x, [len(x[0]), 20, 10, 5, 1])

for i in range(10):
    a = network.forward_pass(x)
    print("a =", a)
    loss_w = BCELoss
    back = network.backward_pass(a, y, loss_function=loss_w)
    print("loss =", back)

for el in network.hidden_layer:
    print("w =", el.w)
    print("b =", el.b)
print("forward_pass =", sigmoid_for_el(network.forward_pass([[-1, -1], [1, 1]])))
