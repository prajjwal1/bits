import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim, lr):
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((hidden_dim,))
        self.w2 = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1,))
        self.lr = lr
        self.cache = {}

    def forward(self, x):
        y1 = x @ self.w1 + self.b1
        z1 = np.maximum(y1, 0)
        y2 = z1 @ self.w2 + self.b2
        z2 = sigmoid(y2)

        self.cache["z1"] = z1
        self.cache["x"] = x
        self.cache["y1"] = y1
        return z2

    def backward(self, dout):
        x, z1 = self.cache["x"], self.cache["z1"]
        y1 = self.cache["y1"]

        dz1 = dout @ self.w2.T
        dw2 = z1.T @ dout
        db2 = dout.sum(axis=0)

        dy1 = dz1 * (y1 > 0)
        dw1 = x.T @ dy1
        dx = dy1 @ self.w1.T
        db1 = dy1.sum(axis=0)
        return dw1, dw2, db1, db2

    def fit(self, x, y, epochs):
        for _ in range(epochs):
            logits = self.forward(x)
            dout = (logits - y.reshape(-1, 1)) / logits.shape[0]
            dw1, dw2, db1, db2 = self.backward(dout)

            self.w1 -= self.lr * dw1
            self.w2 -= self.lr * dw2
            self.b1 -= self.lr * db1
            self.b2 -= self.lr * db2

    def predict(self, x):
        logit = self.forward(x)
        predictions = logit > 0.5
        return predictions.flatten()


# Toy dataset (nonlinear boundary)
X = np.array([[1, 2], [2, 1], [2, 3], [6, 7], [7, 6], [8, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

mlp = TwoLayerMLP(input_dim=2, hidden_dim=16, lr=0.05)
mlp.fit(X, y, epochs=2000)

print(mlp.predict(X))
