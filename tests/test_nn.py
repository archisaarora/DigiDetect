"""Run with:  python -m unittest discover -s tests -v   (numpy only, no pytest needed)"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from digidetect.data import load_bundled, parse_idx  # noqa: E402
from digidetect.nn import MLP  # noqa: E402

train, held = load_bundled()


class NNTests(unittest.TestCase):
    def test_dataset_shapes(self):
        self.assertEqual((train.n, held.n), (5000, 1000))
        self.assertEqual(train.x.shape, (5000, 196))
        self.assertTrue(((train.y >= 0) & (train.y <= 9)).all())
        self.assertTrue(0 <= train.x.min() and train.x.max() <= 1)

    def test_probabilities_sum_to_one(self):
        net = MLP(196, 16, 10, np.random.default_rng(3))
        p = net.forward(train.x[0])[1][0]
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)

    def test_backprop_matches_numerical_gradients(self):
        rng = np.random.default_rng(11)
        net = MLP(196, 12, 10, rng)
        X, Y, eps = train.x[:8], train.y[:8], 1e-3
        net.backward(X, Y)
        for w, g in [("W1", "gW1"), ("b1", "gb1"), ("W2", "gW2"), ("b2", "gb2")]:
            analytic = getattr(net, g).copy()
            arr = getattr(net, w)
            for _ in range(12):
                i = tuple(rng.integers(0, s) for s in arr.shape)
                keep = arr[i]
                arr[i] = keep + eps; up = net.loss(X, Y)
                arr[i] = keep - eps; down = net.loss(X, Y)
                arr[i] = keep
                numeric, mine = (up - down) / (2 * eps), analytic[i] / len(Y)
                self.assertLess(abs(numeric - mine), 1e-3 + 0.01 * abs(numeric), f"{w}{i}")

    def test_training_lowers_loss(self):
        net = MLP(196, 32, 10, np.random.default_rng(5))
        X, Y = train.x[:64], train.y[:64]
        before = net.loss(X, Y)
        for _ in range(30):
            net.train_batch(X, Y, 0.08)
        self.assertLess(net.loss(X, Y), before * 0.5)

    def test_reaches_93_percent_on_held_out(self):
        rng = np.random.default_rng(1)
        net = MLP(196, 64, 10, rng)
        for _ in range(12):
            order = rng.permutation(train.n)
            for b in range(0, train.n, 32):
                i = order[b:b + 32]
                net.train_batch(train.x[i], train.y[i], 0.08)
        acc = net.accuracy(held.x, held.y)
        print(f"held-out accuracy: {acc * 100:.1f}%")
        self.assertGreaterEqual(acc, 0.93)

    def test_save_and_load_roundtrip(self):
        import tempfile, os
        net = MLP(196, 8, 10, np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            net.save(path)
            net2 = MLP.load(path)
        np.testing.assert_array_equal(net.predict(held.x[:50]), net2.predict(held.x[:50]))

    def test_idx_parser_pools_2x2(self):
        img = np.zeros((2, 28, 28), np.uint8); img[0, :2, :2] = 255; img[1, 0, 0] = 255
        images = b"\x00\x00\x08\x03" + (2).to_bytes(4, "big") + (28).to_bytes(4, "big") * 2 + img.tobytes()
        labels = b"\x00\x00\x08\x01" + (2).to_bytes(4, "big") + bytes([3, 7])
        s = parse_idx(images, labels)
        self.assertEqual(s.x.shape, (2, 196))
        self.assertAlmostEqual(float(s.x[0, 0]), 1.0)
        self.assertAlmostEqual(float(s.x[1, 0]), 0.25)
        self.assertEqual(list(s.y), [3, 7])


if __name__ == "__main__":
    unittest.main()
