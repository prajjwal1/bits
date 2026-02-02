import math
from collections import Counter


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def _calc_distance(self, x1, x2):
        dist = 0
        for idx in range(len(x1)):
            dist += (x1[idx] - x2[idx]) ** 2
        return dist

    def _predict_one(self, x):
        res = []
        for idx in range(len(self.x_train)):
            dist = self._calc_distance(x, self.x_train[idx])
            res.append([dist, self.y_train[idx]])
        res.sort(key=lambda x: x[0])
        k_labels = [label for dist, label in res[: self.k]]
        prediction = Counter(k_labels).most_common(1)[0][0]
        return prediction

    def predict(self, x):
        res = []
        for idx in range(len(x)):
            res.append(self._predict_one(x[idx]))
        return res


X_train = [[1, 2], [2, 3], [3, 4], [6, 7]]
y_train = [0, 0, 0, 1]

X_test = [[2, 2], [5, 6]]

knn = KNN(k=3)
knn.fit(X_train, y_train)
print(knn.predict(X_test))  # [0, 0]
