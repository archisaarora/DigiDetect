"""The training loop, as a small stateful object so both the CLI and the GUI can drive it."""
from __future__ import annotations

import numpy as np

from .nn import MLP

BATCH = 32


class Trainer:
    def __init__(self, train, test, hidden=64, epochs=4, lr=0.08, seed=1):
        self.train, self.test = train, test
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.rng = np.random.default_rng(seed)
        self.net = MLP(train.x.shape[1], hidden, 10, self.rng)
        self.order = self.rng.permutation(train.n)
        self.cursor = 0
        self.epoch = 0
        self.step = 0
        self.loss_avg = 0.0                       # smoothed training loss
        self.acc_hist: list[float] = []
        self.loss_hist: list[float] = []

    @property
    def done(self) -> bool:
        return self.epoch >= self.epochs

    def train_batches(self, k: int = 4) -> None:
        """Run up to k mini-batches; record accuracy every 100 steps and at the end."""
        for _ in range(k):
            if self.done:
                return
            end = min(self.cursor + BATCH, self.train.n)
            idx = self.order[self.cursor:end]
            self.cursor = end
            self.step += 1
            loss = self.net.train_batch(self.train.x[idx], self.train.y[idx], self.lr)
            self.loss_avg = self.loss_avg * 0.95 + loss * 0.05 if self.loss_avg else loss
            if self.cursor >= self.train.n:
                self.cursor = 0
                self.epoch += 1
                self.order = self.rng.permutation(self.train.n)
                if self.done:
                    self.measure(final=True)
                    return
            if self.step % 100 == 0:
                self.measure()

    def measure(self, final: bool = False) -> float:
        """Live evaluation uses a 2,000-example slice; the final one uses the whole test split."""
        n = self.test.n if final else min(2000, self.test.n)
        acc = self.net.accuracy(self.test.x[:n], self.test.y[:n])
        self.acc_hist.append(acc)
        self.loss_hist.append(self.loss_avg or 2.3)
        return acc

    def run(self, log=print) -> float:
        """Train to completion (CLI use). Returns final test accuracy."""
        self.measure()
        while not self.done:
            e = self.epoch
            self.train_batches(50)
            if self.epoch != e:
                log(f"epoch {self.epoch}/{self.epochs}  loss {self.loss_avg:.3f}  "
                    f"test accuracy {self.acc_hist[-1] * 100:.1f}%")
        return self.acc_hist[-1]
