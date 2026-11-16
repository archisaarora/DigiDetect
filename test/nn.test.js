/* Run with: npm test   (uses Node's built-in test runner, no dependencies) */
const test = require("node:test");
const assert = require("node:assert");
const path = require("node:path");
globalThis.window = globalThis;
require(path.join(__dirname, "..", "data", "digits.js"));
const { MLP, decode } = require(path.join(__dirname, "..", "js", "nn.js"));

function rng(seed) {
  let s = seed >>> 0;
  const next = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  next.gauss = () => { let u = 0; while (u === 0) u = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * next()); };
  return next;
}
const D = globalThis.DIGITS, train = decode(D.train, D.side), held = decode(D.test, D.side);

test("the dataset decodes to the expected shapes", () => {
  assert.equal(train.n, 5000); assert.equal(held.n, 1000); assert.equal(train.dim, 196);
  assert.equal(train.x.length, 5000 * 196);
  assert.ok(train.y.every((v) => v >= 0 && v <= 9));
});

test("output probabilities are valid and sum to 1", () => {
  const net = new MLP(196, 16, 10, rng(3)), p = net.forward(train.x, 0);
  let sum = 0; for (const v of p) { assert.ok(v >= 0 && v <= 1); sum += v; }
  assert.ok(Math.abs(sum - 1) < 1e-5);
});

test("backpropagation matches numerical gradients", () => {
  const rand = rng(11), net = new MLP(196, 12, 10, rand), idx = [0, 1, 2, 3, 4, 5, 6, 7], eps = 1e-3;   /* small step, so a ReLU rarely flips inside it */
  net.backward(train.x, train.y, idx);
  const checks = [["W1", "gW1"], ["b1", "gb1"], ["W2", "gW2"], ["b2", "gb2"]];
  for (const [w, g] of checks) {
    const analytic = Float32Array.from(net[g]);            /* copy: backward() is not called again, but be safe */
    for (let trial = 0; trial < 12; trial++) {
      const i = Math.floor(rand() * net[w].length), keep = net[w][i];
      net[w][i] = keep + eps; const up = net.loss(train.x, train.y, idx);
      net[w][i] = keep - eps; const down = net.loss(train.x, train.y, idx);
      net[w][i] = keep;
      const numeric = (up - down) / (2 * eps), mine = analytic[i] / idx.length;
      assert.ok(Math.abs(numeric - mine) < 1e-3 + 0.01 * Math.abs(numeric), `${w}[${i}]: numeric ${numeric.toFixed(5)} vs backprop ${mine.toFixed(5)}`);
    }
  }
});

test("training lowers the loss", () => {
  const net = new MLP(196, 32, 10, rng(5)), idx = Array.from({ length: 64 }, (_, i) => i);
  const before = net.loss(train.x, train.y, idx);
  for (let k = 0; k < 30; k++) net.trainBatch(train.x, train.y, idx, 0.08);
  assert.ok(net.loss(train.x, train.y, idx) < before * 0.5);
});

test("reaches at least 93% on 1,000 held-out digits after 12 passes", () => {
  const rand = rng(1), net = new MLP(196, 64, 10, rand), order = Array.from({ length: train.n }, (_, i) => i);
  for (let epoch = 0; epoch < 12; epoch++) {
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let b = 0; b < train.n; b += 32) net.trainBatch(train.x, train.y, order.slice(b, b + 32), 0.08);
  }
  const acc = net.accuracy(held.x, held.y, held.n);
  console.log(`held-out accuracy: ${(acc * 100).toFixed(1)}%`);
  assert.ok(acc >= 0.93);
});
