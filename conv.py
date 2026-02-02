import numpy as np


def conv1d(x, w):
    N = len(x)
    K = len(w)
    assert N - K + 1 > 0

    y = np.zeros(N - K + 1)

    for i in range(N - K + 1):
        y[i] = np.dot(x[i : i + K], w)
    return y


def conv2d(x, w):
    H, W = x.shape
    kH, kW = w.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    assert H - kH + 1 > 0
    assert W - kW + 1 > 0

    x = np.stack(
        [
            x[i : i + kH, j : j + kW].reshape(-1)
            for i in range(out_H)
            for j in range(out_W)
        ]
    )
    out = x @ w.reshape(-1)
    return out.reshape(out_H, out_W)


from itertools import product


def conv_nd(x, w):
    x_shape = x.shape
    w_shape = w.shape

    out_shape = tuple((x_dim - w_dim + 1) for x_dim, w_dim in zip(x_shape, w_shape))

    if any(d <= 0 for d in out_shape):
        return []

    positions = list(product(*[range(d) for d in out_shape]))
    X = np.stack(
        [
            x[tuple(slice(p, p + k) for p, k in zip(pos, w_shape))].reshape(-1)
            for pos in positions
        ]
    )
    out = X @ w.reshape(-1)
    return out.reshape(out_shape)


# def conv_nd(x, w):
#     ndim = x.ndim
#     out_shape = [x.shape[i] - w.shape[i] + 1 for i in range(ndim)]
#     if any(d <= 0 for d in out_shape):
#         return np.array([])

#     y = np.zeros(out_shape)
#     for out_idx in np.ndindex(*out_shape):
#         slices = tuple(
#             slice(out_idx[dim], out_idx[dim] + w.shape[dim]) for dim in range(ndim)
#         )
#         # print(x[slices])
#         y[out_idx] = np.sum(x[slices] * w)
#     return y


# x = [1, 2, 3, 4]
# w = [1, 0, -1]
# print(conv1d(x, w))

# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# w = np.array([[1, 0], [0, -1]])
# print(conv2d(x, w))

x = np.random.randn(5, 5, 5)
w = np.random.randn(3, 3, 3)
print(conv_nd(x, w))
# print(x[(slice(0, 2, None), slice(0, 2, None))])
