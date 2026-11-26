"""Fuzzy C-Means + Takagi-Sugeno-Kang orden 1 para clasificación de gestos sEMG.

    y(x) = sum_i w_i(x) * (P_i @ x + q_i) / sum_i w_i(x)

cada regla i es un "especialista local" con 5 salidas (una por gesto).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from config import FUZZIFIER, MAX_ITER, N_CLUSTERS, RIDGE, TOL


def partition_coefficient(u: np.ndarray) -> float:
    """PC = (1/n) sum_i sum_k u_ik^2  (media POR FILA, no por elemento)."""
    u = np.atleast_2d(u)
    return float(u.shape[1] * np.mean(u**2))


def partition_entropy(u: np.ndarray) -> float:
    """PE = -(1/n) sum_i sum_k u_ik ln(u_ik)."""
    u = np.atleast_2d(u)
    return float(-u.shape[1] * np.mean(u * np.log(np.clip(u, 1e-12, 1.0))))


class FuzzyCMeans:
    """FCM de Bezdek con centros inicializados por k-means (semilla fija)."""

    def __init__(self, c=N_CLUSTERS, m=FUZZIFIER, max_iter=MAX_ITER, tol=TOL, seed=13):
        self.c, self.m, self.max_iter, self.tol, self.seed = c, m, max_iter, tol, seed
        self.centers = None
        self.U = None
        self.n_iter = 0
        self.objective = []

    def _distances(self, X):
        d = np.sum((X[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        return np.maximum(d, 1e-12)

    def _membership(self, X):
        u = 1.0 / self._distances(X) ** (1.0 / (self.m - 1.0))
        return u / u.sum(axis=1, keepdims=True)

    def fit(self, X, init_centers=None):
        """init_centers: (c, d). Si se da, se usan prototipos por clase (reglas interpretables)."""
        X = np.asarray(X, dtype=np.float64)
        if init_centers is None:
            self.centers = KMeans(n_clusters=self.c, n_init=10, random_state=self.seed).fit(X).cluster_centers_
        else:
            self.centers = np.asarray(init_centers, dtype=np.float64).copy()
        U = self._membership(X)
        for it in range(self.max_iter):
            um = U**self.m
            self.centers = (um.T @ X) / um.sum(axis=0)[:, None]
            self.objective.append(float(np.sum(um * self._distances(X))))
            U_new = self._membership(X)
            delta = np.max(np.abs(U_new - U))
            U = U_new
            self.n_iter = it + 1
            if delta < self.tol:
                break
        self.U = U
        return self

    def membership(self, X):
        return self._membership(np.asarray(X, dtype=np.float64))

    def gaussian_membership(self, X):
        """Alternativa tipo ANFIS: gaussianas diagonales con dispersión ponderada."""
        X = np.asarray(X, dtype=np.float64)
        um = self.U**self.m
        w = um.sum(axis=0)
        means = (um.T @ X) / np.maximum(w[:, None], 1e-12)
        var = np.maximum((um.T @ (X**2)) / np.maximum(w[:, None], 1e-12) - means**2, 1e-6)
        g = np.exp(-0.5 * np.sum((X[:, None, :] - means) ** 2 / var, axis=2))
        return g / g.sum(axis=1, keepdims=True)


class TSKClassifier:
    """c consecuentes lineales; cada uno produce un vector de 5 puntajes."""

    def __init__(self, ridge=RIDGE):
        self.ridge = ridge
        self.P = None  # (c, 5, 32)
        self.q = None  # (c, 5)

    def fit(self, X, Y_onehot, U, m=FUZZIFIER):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y_onehot, dtype=np.float64)
        A = np.hstack([X, np.ones((len(X), 1))])
        c, n_out, n_feat = U.shape[1], Y.shape[1], X.shape[1]
        self.P, self.q = np.zeros((c, n_out, n_feat)), np.zeros((c, n_out))
        for i in range(c):
            w = np.sqrt(np.maximum(U[:, i], 1e-12) ** m)[:, None]
            reg = self.ridge * np.eye(n_feat + 1)
            reg[-1, -1] = 0
            theta = np.linalg.solve((A * w).T @ (A * w) + reg, (A * w).T @ (Y * w))
            self.P[i], self.q[i] = theta[:-1].T, theta[-1]
        return self

    def scores(self, X, W):
        X, W = np.asarray(X, dtype=np.float64), np.asarray(W, dtype=np.float64)
        num = np.zeros((len(X), self.P.shape[1]), dtype=np.float64)
        for i in range(len(self.P)):
            num += W[:, i, None] * (X @ self.P[i].T + self.q[i])
        return num / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X, W):
        return np.argmax(self.scores(X, W), axis=1)
