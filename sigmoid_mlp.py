# binary classification
# 1 logit, sigmoid, binary cross entropy, y \in {0, 1}
import numpy as np


def init_layer(fan_in, fan_out):
    std_dev = np.sqrt(1 / fan_in)
    return np.random.normal(size=(fan_in, fan_out)) * std_dev

def relu(x):
    return np.maximum(0, x)

def relu_backward(x):
    return (x > 0)


class MLP:
    def __init__(self, dims):
        self.w1 = init_layer(dims[0], dims[1])
        self.b1 = np.zeros(
            (dims[1]),
        )
        self.w2 = init_layer(dims[1], dims[2])
        self.b2 = np.zeros(
            dims[2],
        )
        self.cache = {}

    def forward(self, x):
        y1 = x @ self.w1 + self.b1
        z1 = relu(y1)
        y2 = z1 @ self.w2 + self.b2
        self.cache["x"] = x
        self.cache["z1"] = z1
        self.cache["y1"] = y1
        return y2

    def backward(self, dout):
        x = self.cache["x"]
        z1 = self.cache["z1"]
        y1 = self.cache["y1"]

        dz1 = dout @ self.w2.T
        dw2 = z1.T @ dout

        db2 = dout.sum(axis=0)
        g = relu_backward(y1) * dz1
        dx = g @ self.w1.T
        dw1 = x.T @ g
        db1 = g.sum(axis=0)
        db = dout.sum(axis=0)
        return dw1, dw2, db1, db2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(logits, y, eps=1e-5):
    probs = sigmoid(logits)
    loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    return probs, loss


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32)
y = np.array([0, 1, 1, 0]).reshape(-1, 1)

dims = [2, 4, 1]
model = MLP(dims)

for _ in range(100):
    logits = model.forward(x)  # [bs, dims[1]]
    probs, loss = binary_cross_entropy(logits, y)
    dout = (probs - y) / logits.shape[0]
    dw1, dw2, db1, db2 = model.backward(dout)
    model.w1 -= dw1
    model.b1 -= db1
    model.w2 -= dw2
    model.b2 -= db2
    print(f"loss: {loss}")
