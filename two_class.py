import numpy as np


def init_layer(fan_in, fan_out):
    std_dev = np.sqrt(1 / fan_in)
    return np.random.randn(fan_in, fan_out) * std_dev


def relu(x):
    return np.maximum(0, x)


class MLP:
    def __init__(self, dims):
        self.w1 = init_layer(dims[0], dims[1])
        self.b1 = np.zeros((1, dims[1]))
        self.w2 = init_layer(dims[1], dims[2])
        self.b2 = np.zeros((1, dims[2]))
        self.cache = {}

    def forward(self, x):
        y1 = x @ self.w1 + self.b1
        z1 = relu(y1)
        y2 = z1 @ self.w2 + self.b2
        self.cache["z1"] = z1
        self.cache["x"] = x
        self.cache["y1"] = y1
        return y2

    def backward(self, dout):
        z1, x, y1 = self.cache["z1"], self.cache["x"], self.cache["y1"]
        dz1 = dout @ self.w2.T
        dw2 = z1.T @ dout
        db2 = dout.sum(axis=0)

        dy1 = (y1 > 0) * dz1
        dx = dy1 @ self.w1.T
        dw1 = x.T @ dy1
        db1 = dy1.sum(axis=0)
        return dw2, db2, dw1, db1


x = np.array(
    [
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1],
    ]
)
y = np.array([0, 1, 1, 0])

dims = [2, 4, 2]
model = MLP(dims)
num_classes = 2


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def cross_entropy(logits, y, eps=1e-5):
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(len(logits)), y] + eps))
    return probs, loss


for _ in range(20):
    logits = model.forward(x)
    probs, loss = cross_entropy(logits, y)
    dout = (probs - one_hot(y, num_classes=2)) / logits.shape[0]

    dw2, db2, dw1, db1 = model.backward(dout)
    print(loss)

    model.w2 -= dw2
    model.w1 -= dw1
    model.b1 -= db1
    model.b2 -= db2


# import numpy as np

# def init_layer(fan_in, fan_out):
#     std_dev = np.sqrt(1 / fan_in)
#     return np.random.normal(size=(fan_in, fan_out)) * std_dev


# def relu(x):
#     return np.maximum(0, x)

# def relu_backward(x):
#     return (x > 0)


# class MLP:
#     def __init__(self, dims):
#         self.w1 = init_layer(dims[0], dims[1])
#         self.b1 = np.zeros(
#             (dims[1]),
#         )
#         self.w2 = init_layer(dims[1], dims[2])
#         self.b2 = np.zeros(
#             dims[2],
#         )
#         self.cache = {}

#     def forward(self, x):
#         y1 = x @ self.w1 + self.b1
#         z1 = relu(y1)
#         y2 = z1 @ self.w2 + self.b2
#         self.cache["x"] = x
#         self.cache["z1"] = z1
#         self.cache["y1"] = y1
#         return y2

#     def backward(self, dout):
#         x = self.cache["x"]
#         z1 = self.cache["z1"]
#         y1 = self.cache["y1"]

#         dz1 = dout @ self.w2.T
#         dw2 = z1.T @ dout

#         db2 = dout.sum(axis=0)
#         g = relu_backward(y1) * dz1
#         dx = g @ self.w1.T
#         dw1 = x.T @ g
#         db1 = g.sum(axis=0)
#         db = dout.sum(axis=0)
#         return dw1, dw2, db1, db2

# dims = [2, 4, 2]
# model = MLP(dims)

# x = np.array([
#     [0, 1],
#     [1, 0],
#     [0, 0],
#     [1, 1],
# ])
# y = np.array([0, 1, 1, 0])

# def one_hot(y, num_classes):
#     return np.eye(num_classes)[y]

# def softmax(x):
#     exp = np.exp(x - x.max(axis=-1, keepdims=True))
#     return exp / exp.sum(axis=-1, keepdims=True)

# def cross_entropy(logits):
#     probs = softmax(logits)
#     loss = -np.mean(np.log(probs[np.arange(len(logits)), y]))
#     return probs, loss

# for _ in range(10):
#     logits = model.forward(x)
#     probs, loss = cross_entropy(logits)
#     dout = (probs - one_hot(y, 2)) / logits.shape[0]
#     dw1, dw2, db1, db2 = model.backward(dout)

#     model.w1 -= dw1
#     model.w2 -= dw2
#     model.b1 -= db1
#     model.b2 -= db2
#     print(f"loss: {loss}")
