import numpy as np


def relu(x):
    return np.where(x > 0, x, 0.01 * x)


def init_layer(fan_in, fan_out):
    std_dev = np.sqrt(2 / fan_in)
    return np.random.normal(size=(fan_in, fan_out)) * std_dev


class Model:
    def __init__(self, dims):
        self.w1 = init_layer(dims[0], dims[1])
        self.b1 = np.full(dims[1], 0.1)
        self.w2 = init_layer(dims[1], dims[2])
        self.b2 = np.full(dims[2], 0.1)
        self.cache = {}

    def forward(self, x):
        y1 = x @ self.w1 + self.b1
        z1 = relu(y1)
        y2 = z1 @ self.w2 + self.b2
        self.cache["z1"] = z1
        self.cache["x"] = x
        return y2

    def backward(self, dout):
        z1 = self.cache["z1"]
        x = self.cache["x"]

        dw2 = z1.T @ dout
        dz1 = dout @ self.w2.T
        db2 = dout.sum(axis=0)

        dy1 = np.where(z1 > 0, 1, 0.01) * dz1
        dw1 = x.T @ dy1
        dx = dy1 @ self.w1.T
        db1 = dy1.sum(axis=0)

        return dw1, dw2, db1, db2


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits, y):
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(len(logits)), y]))
    return probs, loss


def one_hot(num_classes, y):
    return np.eye(num_classes)[y]


x = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
num_classes = 2

model = Model([2, 4, 2])

for _ in range(20):
    logits = model.forward(x)
    probs, loss = cross_entropy(logits, y)
    dout = (probs - one_hot(num_classes, y)) / len(x)

    print(f"loss: {loss}")

    dw1, dw2, db1, db2 = model.backward(dout)
    model.w1 -= dw1
    model.b1 -= db1
    model.w2 -= dw2
    model.b2 -= db2
