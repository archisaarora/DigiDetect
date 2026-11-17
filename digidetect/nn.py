"""A small neural network written from scratch: one hidden layer (ReLU), a softmax
output, cross-entropy loss, and mini-batch gradient descent with momentum.

Only numpy is used, and only for array maths: forward pass, backpropagation and the
weight updates are all written out by hand (no autograd, no ML framework).
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, n_hid: int, n_out: int, rng: np.random.Generator):
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        # He initialisation
        self.W1 = (rng.standard_normal((n_hid, n_in)) * np.sqrt(2 / n_in)).astype(np.float32)
        self.b1 = np.zeros(n_hid, np.float32)
        self.W2 = (rng.standard_normal((n_out, n_hid)) * np.sqrt(2 / n_hid)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        # momentum buffers and most recent gradients
        self.params = ["W1", "b1", "W2", "b2"]
        for name in self.params:
            w = getattr(self, name)
            setattr(self, "v" + name, np.zeros_like(w))
            setattr(self, "g" + name, np.zeros_like(w))

    # ---- forward ---------------------------------------------------------------
    def forward(self, x: np.ndarray):
        """x: (n_in,) or (batch, n_in). Returns (hidden activations, class probabilities)."""
        x = np.atleast_2d(x)
        h = np.maximum(0, x @ self.W1.T + self.b1)             # ReLU
        z = h @ self.W2.T + self.b2
        z = z - z.max(axis=1, keepdims=True)                   # stable softmax
        e = np.exp(z)
        return h, e / e.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[1].argmax(axis=1)

    # ---- loss / backprop ---------------------------------------------------------
    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Mean cross-entropy on (X, Y), without changing anything."""
        p = self.forward(X)[1]
        return float(-np.log(p[np.arange(len(Y)), Y] + 1e-9).mean())

    def backward(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Backpropagation. Fills gW1, gb1, gW2, gb2 with gradients *summed* over the
        examples in the batch (the same convention as the JavaScript version) and
        returns the mean loss."""
        n = len(Y)
        h, p = self.forward(X)
        loss = float(-np.log(p[np.arange(n), Y] + 1e-9).mean())

        d = p.copy()
        d[np.arange(n), Y] -= 1                                # dLoss/dlogit for softmax + cross-entropy
        self.gW2 = d.T @ h
        self.gb2 = d.sum(axis=0)
        dh = d @ self.W2
        dh *= h > 0                                            # ReLU passes gradient only where it was active
        self.gW1 = dh.T @ np.atleast_2d(X)
        self.gb1 = dh.sum(axis=0)
        return loss

    # ---- training ----------------------------------------------------------------
    def train_batch(self, X, Y, lr: float, momentum: float = 0.9) -> float:
        """One gradient step with momentum. Returns the mean loss."""
        loss = self.backward(X, Y)
        s = lr / len(Y)
        for name in self.params:
            v = getattr(self, "v" + name)
            v *= momentum
            v -= s * getattr(self, "g" + name)
            getattr(self, name).__iadd__(v)
        return loss

    def accuracy(self, X, Y) -> float:
        return float((self.predict(X) == Y).mean())

    # ---- saving ------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "MLP":
        d = np.load(path)
        net = cls(d["W1"].shape[1], d["W1"].shape[0], d["W2"].shape[0], np.random.default_rng(0))
        net.W1, net.b1, net.W2, net.b2 = (d[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
        return net
