/* DigiDetect: an MLP trained from scratch in the browser on MNIST. */
Site.registerDemo("digidetect", function (root) {
  const { h, fit, loop, colors } = Site;
  const BATCH = 32;
  let data = null, net = null, order = [], cursor = 0, epoch = 0, step = 0;
  let training = false, hidden = 64, lr = 0.08, epochs = 4;
  let accHist = [], lossHist = [], lossAvg = 0, seed = 1, ready = false;
  let datasetName = "Bundled MNIST sample", fullDatasetLoaded = false;

  const chart = h("canvas", { style: "height:170px", role: "img", "aria-label": "Model accuracy and training loss" });
  const tiles = h("canvas", { style: "height:110px", role: "img", "aria-label": "Visualizations of learned hidden-layer weights" });
  const pad = h("canvas.sc-pad", { width: 224, height: 224, role: "img", "aria-label": "Draw a digit from zero to nine", tabindex: "0" });
  const bars = h("div.sc-bars");
  const barEls = Array.from({ length: 10 }, (_, d) => {
    const fill = h("i");
    const row = h("div.sc-bar", null, h("span", { text: d }), h("div", null, fill));
    bars.append(row); return { row, fill };
  });
  const verdict = h("p.sc-verdict", { "aria-live": "polite", text: "Draw a digit to make a prediction" });
  const stat = (label, initial = "—") => {
    const b = h("b", { text: initial });
    return { el: h("div.d-stat", null, b, h("span", { text: label })), b };
  };
  const sAcc = stat("validation accuracy"), sEpoch = stat("epochs completed", "0"), sLoss = stat("recent training loss");
  const sCount = stat("training examples", "5,000");
  const status = h("p.sc-status", { text: "Preparing the dataset…" });
  const toggle = h("button.d-btn.solid", { type: "button", text: "Pause training", onclick: () => setTraining(!training) });
  const fullBtn = h("button.d-btn", { type: "button", text: "Load full MNIST (60k train + 10k test)", onclick: loadFullDataset });
  const quickBtn = h("button.d-btn", { type: "button", text: "Use quick sample", onclick: loadQuickDataset });
  const hidSel = h("select", { "aria-label": "Hidden layer size", onchange: (e) => { hidden = +e.target.value; if (ready) reset(); } },
    [32, 64, 128].map((n) => h("option", { value: n, text: `${n} hidden units`, selected: n === 64 })));
  const epochSel = h("select", { "aria-label": "Training epochs", onchange: (e) => { epochs = +e.target.value; if (ready) reset(); } },
    [2, 4, 6, 8].map((n) => h("option", { value: n, text: `${n} epochs`, selected: n === 4 })));

  root.append(
    h("div.sc-topline", null, h("div", null, h("span.sc-kicker", { text: "LIVE EXPERIMENT" }), status),
      h("span.sc-dataset", { text: datasetName })),
    h("div.sc-layout", null,
      h("section.sc-panel.sc-training", null,
        h("div.sc-panel-head", null, h("div", null, h("h3", { text: "Training lab" }), h("p", { text: "Watch a small neural network learn from labeled examples." })),
          h("span.sc-live", null, h("i"), " IN-BROWSER")),
        chart,
        h("div.d-stats", null, sAcc.el, sEpoch.el, sLoss.el, sCount.el),
        h("div.d-row", null, toggle, fullBtn, quickBtn),
        h("div.d-row.sc-controls", null, h("label.d-field", null, "Model size ", hidSel), h("label.d-field", null, "Training length ", epochSel)),
        h("div.sc-weights", null, h("div.sc-subhead", null, h("strong", { text: "What the hidden layer learns" }), h("span", { text: "Each square visualizes one neuron's pixel weights" })), tiles)),
      h("section.sc-panel.sc-draw", null,
        h("div.sc-panel-head", null, h("div", null, h("h3", { text: "Try it yourself" }), h("p", { text: "Draw a number and see the model's ten class scores." }))),
        h("div.sc-draw-area", null, pad, h("div.d-row", null,
          h("button.d-btn", { type: "button", text: "Clear canvas", onclick: clearPad }),
          h("button.d-btn", { type: "button", text: "Surprise me", onclick: drawExample }))),
        verdict, bars,
        h("p.sc-footnote", { text: "Predictions are model probabilities, not guarantees. The model has only seen MNIST-style handwriting." })))
  );

  function setDatasetLabel(text) {
    const el = root.querySelector(".sc-dataset");
    if (el) el.textContent = text;
  }
  function setTraining(on) {
    training = !!on && epoch < epochs;
    toggle.textContent = training ? "Pause training" : epoch >= epochs ? "Training complete" : "Resume training";
    toggle.disabled = epoch >= epochs;
    status.textContent = training ? `Training ${datasetName.toLowerCase()} in this browser tab` : epoch >= epochs ? "Training finished — draw a digit or change the experiment settings." : "Training paused";
  }

  function decodeBundled() {
    if (!window.DIGITS) throw new Error("Bundled sample data is missing.");
    return { train: NN.decode(DIGITS.train, DIGITS.side), test: NN.decode(DIGITS.test, DIGITS.side) };
  }

  // Public mirrors are tried file-by-file. GitHub raw and jsDelivr are first
  // because they support browser CORS; Google Cloud and the PyTorch mirror
  // remain fallbacks. Both gzip and transparently decompressed IDX are accepted.
  const MNIST_FILES = [
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
  ];
  const MNIST_MIRRORS = [
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://cdn.jsdelivr.net/gh/fgnt/mnist@master/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/"
  ];

  async function fetchGzip(url) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45000);
    try {
      const response = await fetch(url, {
        mode: "cors", credentials: "omit", cache: "no-store", signal: controller.signal
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      let bytes = new Uint8Array(await response.arrayBuffer());
      // Fetch transparently decompresses HTTP Content-Encoding:gzip. Files
      // stored as .gz on GitHub arrive as gzip bytes and need explicit inflate.
      if (bytes[0] === 0 && bytes[1] === 0 && bytes[2] === 8) return bytes;
      if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
        throw new Error("The server returned something other than an MNIST gzip/IDX file (often a blocked request or HTML error page).");
      }
      if (typeof DecompressionStream === "undefined") {
        throw new Error("This browser cannot decompress gzip files. Update your browser or use the bundled sample.");
      }
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      bytes = new Uint8Array(await new Response(stream).arrayBuffer());
      if (!(bytes[0] === 0 && bytes[1] === 0 && bytes[2] === 8)) throw new Error("The downloaded file was not a valid MNIST IDX file");
      return bytes;
    } catch (err) {
      if (err.name === "AbortError") throw new Error("The download timed out after 45 seconds");
      throw err;
    } finally { clearTimeout(timeout); }
  }

  // Resolve each file independently. A mirror can host some files but not
  // others, so failing the entire four-file download on one 404 is avoidable.
  async function downloadFullMNIST() {
    const resolved = [];
    for (let fileIndex = 0; fileIndex < MNIST_FILES.length; fileIndex++) {
      const filename = MNIST_FILES[fileIndex];
      let payload = null;
      const errors = [];
      for (let mirrorIndex = 0; mirrorIndex < MNIST_MIRRORS.length; mirrorIndex++) {
        const url = MNIST_MIRRORS[mirrorIndex] + filename;
        status.textContent = `Downloading ${filename} · source ${mirrorIndex + 1}/${MNIST_MIRRORS.length}`;
        try {
          payload = await fetchGzip(url);
          break;
        } catch (err) {
          errors.push(`${new URL(url).host}: ${err.message}`);
        }
      }
      if (!payload) throw new Error(`Could not download ${filename}. ${errors.join(" | ")}`);
      resolved.push(payload);
    }
    return resolved;
  }

  function readU32(bytes, offset) {
    return ((bytes[offset] * 0x1000000) + (bytes[offset + 1] << 16) + (bytes[offset + 2] << 8) + bytes[offset + 3]) >>> 0;
  }

  function parseIDX(images, labels) {
    const count = readU32(images, 4), rows = readU32(images, 8), cols = readU32(images, 12);
    const labelCount = readU32(labels, 4);
    if (count !== labelCount || rows !== 28 || cols !== 28) throw new Error("The downloaded MNIST files have an unexpected format.");
    const x = new Float32Array(count * 196), y = labels.slice(8, 8 + count);
    // Average-pool each 2x2 block: preserve the existing compact 14x14 model input.
    for (let n = 0; n < count; n++) {
      const src = 16 + n * 784, dst = n * 196;
      for (let r = 0; r < 14; r++) for (let c = 0; c < 14; c++) {
        const i = src + (r * 2) * 28 + c * 2;
        x[dst + r * 14 + c] = (images[i] + images[i + 1] + images[i + 28] + images[i + 29]) / (4 * 255);
      }
    }
    return { n: count, x, y, dim: 196 };
  }

  async function loadFullDataset() {
    fullBtn.disabled = true; quickBtn.disabled = true;
    status.textContent = "Connecting to a public MNIST mirror…";
    setDatasetLabel("Loading full MNIST");
    try {
      status.textContent = "Trying public MNIST sources. Each file can use a different mirror…";
      const [trainImages, trainLabels, testImages, testLabels] = await downloadFullMNIST();
      status.textContent = "Download complete. Checking all four image and label files…";
      data = { train: parseIDX(trainImages, trainLabels), test: parseIDX(testImages, testLabels) };
      if (data.train.n !== 60000 || data.test.n !== 10000) throw new Error("The downloaded dataset did not contain the expected 60,000 training and 10,000 test images.");
      datasetName = "Full MNIST · 70,000 images"; fullDatasetLoaded = true;
      sCount.b.textContent = "60,000"; setDatasetLabel(datasetName);
      status.textContent = "Full MNIST is ready. Starting a fresh model…";
      reset();
    } catch (err) {
      status.textContent = `Full MNIST could not be loaded: ${err.message} The bundled 6,000-image sample is still available. Run from localhost or GitHub Pages, not file://.`;
      setDatasetLabel("Bundled sample · offline ready");
    } finally { fullBtn.disabled = false; quickBtn.disabled = false; }
  }

  function loadQuickDataset() {
    data = decodeBundled(); datasetName = "MNIST sample · 6,000 images"; fullDatasetLoaded = false;
    sCount.b.textContent = "5,000"; setDatasetLabel(datasetName); reset();
  }

  function reset() {
    const rand = Site.rng(1000 + seed++);
    net = new NN.MLP(data.train.dim, hidden, 10, rand);
    order = Array.from({ length: data.train.n }, (_, i) => i); shuffle(rand);
    cursor = 0; epoch = 0; step = 0; accHist = []; lossHist = []; lossAvg = 0; net._rand = rand;
    ready = true; measure(false); setTraining(true); classify();
  }
  function shuffle(rand) { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } }

  function measure(final = false) {
    // Live evaluation uses a 2,000-example slice to keep the interface responsive.
    // At the end, evaluate against the complete held-out test split.
    const n = final ? data.test.n : Math.min(2000, data.test.n);
    const acc = net.accuracy(data.test.x, data.test.y, n);
    accHist.push(acc); lossHist.push(lossAvg || 2.3);
    sAcc.b.textContent = `${(acc * 100).toFixed(1)}%${final ? " · full test" : ""}`;
    sEpoch.b.textContent = `${epoch} / ${epochs}`;
    sLoss.b.textContent = (lossAvg || 2.3).toFixed(3);
  }

  function trainSome() {
    for (let k = 0; k < 4 && training; k++) {
      const end = Math.min(cursor + BATCH, data.train.n);
      const idx = order.slice(cursor, end); cursor = end; step++;
      const loss = net.trainBatch(data.train.x, data.train.y, idx, lr);
      lossAvg = lossAvg ? lossAvg * 0.95 + loss * 0.05 : loss;
      if (cursor >= data.train.n) {
        cursor = 0; epoch++; shuffle(net._rand);
        if (epoch >= epochs) { measure(true); setTraining(false); }
      }
      if (step % 100 === 0 && training) measure(false);
    }
  }

  function drawChart() {
    const { ctx, w, h: hh } = fit(chart); ctx.clearRect(0, 0, w, hh);
    const total = Math.max(12, Math.ceil((data.train.n / BATCH) * epochs / 100) + 1);
    const X = (i) => 38 + (i / total) * (w - 52), Y = (v) => hh - 22 - v * (hh - 38);
    ctx.font = "11px ui-monospace, monospace"; ctx.fillStyle = "#718096"; ctx.strokeStyle = "#e6eaf0"; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 0.9, 1]) { ctx.beginPath(); ctx.moveTo(38, Y(v)); ctx.lineTo(w - 10, Y(v)); ctx.stroke(); ctx.fillText(`${Math.round(v * 100)}%`, 3, Y(v) + 4); }
    const line = (arr, color, scale) => { if (!arr.length) return; ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath(); arr.forEach((v, i) => { const y = Y(Math.min(1, v * scale)); i ? ctx.lineTo(X(i), y) : ctx.moveTo(X(i), y); }); ctx.stroke(); };
    line(lossHist, "#a0aaa3", 1 / 2.4); line(accHist, "#477b5b", 1);
    ctx.fillStyle = "#477b5b"; ctx.fillText("accuracy", Math.max(44, w - 84), 14); ctx.fillStyle = "#7a857d"; ctx.fillText("loss", Math.max(44, w - 42), 14);
  }

  function drawTiles() {
    const { ctx, w, h: hh } = fit(tiles); ctx.clearRect(0, 0, w, hh);
    const cols = 12, rows = 2, side = 14, cell = Math.floor(Math.min((w - (cols - 1) * 5) / cols, (hh - 5) / rows)), px = cell / side;
    for (let u = 0; u < cols * rows && u < net.nHid; u++) {
      const x0 = (u % cols) * (cell + 5), y0 = Math.floor(u / cols) * (cell + 5), r = u * net.nIn;
      let m = 1e-6; for (let i = 0; i < net.nIn; i++) m = Math.max(m, Math.abs(net.W1[r + i]));
      for (let i = 0; i < net.nIn; i++) {
        const v = net.W1[r + i] / m;
        ctx.fillStyle = v > 0 ? `rgba(71,123,91,${Math.abs(v).toFixed(2)})` : `rgba(145,155,147,${Math.abs(v).toFixed(2)})`;
        ctx.fillRect(x0 + (i % side) * px, y0 + Math.floor(i / side) * px, px + 0.5, px + 0.5);
      }
    }
  }

  const pctx = pad.getContext("2d"), small = document.createElement("canvas"); small.width = small.height = 14;
  let drawing = false, last = null, hasInk = false;
  function clearPad() {
    pctx.fillStyle = "#27372d"; pctx.fillRect(0, 0, 224, 224); hasInk = false;
    verdict.textContent = "Draw a digit to make a prediction";
    barEls.forEach((b) => { b.fill.style.width = "0"; b.row.classList.remove("top"); });
  }
  function drawExample() {
    if (!data) return;
    const i = Math.floor(Math.random() * data.test.n), x = data.test.x.subarray(i * 196, (i + 1) * 196);
    const image = pctx.createImageData(224, 224);
    for (let y = 0; y < 224; y++) for (let xx = 0; xx < 224; xx++) {
      const v = Math.round(x[Math.floor(y / 16) * 14 + Math.floor(xx / 16)] * 255), off = (y * 224 + xx) * 4;
      image.data[off] = v; image.data[off + 1] = v; image.data[off + 2] = v; image.data[off + 3] = 255;
    }
    pctx.putImageData(image, 0, 0); hasInk = true; classify();
    verdict.textContent = `${verdict.textContent} · actual label: ${data.test.y[i]}`;
  }
  const pos = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * 224 / r.width, (e.clientY - r.top) * 224 / r.height]; };
  pad.addEventListener("pointerdown", (e) => { drawing = true; last = pos(e); pad.setPointerCapture(e.pointerId); stroke(last); });
  pad.addEventListener("pointermove", (e) => { if (drawing) stroke(pos(e)); });
  const end = () => { drawing = false; }; pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
  function stroke(p) {
    pctx.strokeStyle = "#f7f8fa"; pctx.lineWidth = 17; pctx.lineCap = pctx.lineJoin = "round";
    pctx.beginPath(); pctx.moveTo(last[0], last[1]); pctx.lineTo(p[0] + 0.01, p[1]); pctx.stroke(); last = p; hasInk = true; classify();
  }
  function classify() {
    if (!hasInk || !net) return;
    const img = pctx.getImageData(0, 0, 224, 224).data;
    let x0 = 224, y0 = 224, x1 = 0, y1 = 0, mx = 0, my = 0, mass = 0;
    for (let y = 0; y < 224; y += 2) for (let x = 0; x < 224; x += 2) {
      const v = img[(y * 224 + x) * 4] - 23; if (v < 60) continue;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); mx += x * v; my += y * v; mass += v;
    }
    if (!mass) return;
    const scale = 10 / Math.max(x1 - x0, y1 - y0, 20), sctx = small.getContext("2d");
    sctx.fillStyle = "#000"; sctx.fillRect(0, 0, 14, 14); sctx.imageSmoothingEnabled = true; sctx.imageSmoothingQuality = "high";
    sctx.setTransform(scale, 0, 0, scale, 7 - (mx / mass) * scale, 7 - (my / mass) * scale); sctx.drawImage(pad, 0, 0); sctx.setTransform(1, 0, 0, 1, 0, 0);
    const d = sctx.getImageData(0, 0, 14, 14).data, x = new Float32Array(196);
    for (let i = 0; i < 196; i++) x[i] = Math.max(0, (d[i * 4] - 23) / 232);
    const p = net.forward(x); let best = 0;
    for (let k = 1; k < 10; k++) if (p[k] > p[best]) best = k;
    barEls.forEach((b, k) => { b.fill.style.width = `${(p[k] * 100).toFixed(1)}%`; b.row.classList.toggle("top", k === best); });
    verdict.textContent = `Prediction: ${best} · ${(p[best] * 100).toFixed(1)}% model confidence`;
  }

  clearPad();
  let paint = 0;
  const run = loop((dt) => {
    if (!ready) return;
    if (training) trainSome();
    paint += dt;
    if (paint > 0.14) { paint = 0; drawChart(); drawTiles(); if (training) classify(); }
  });
  return {
    start() {
      if (!data) {
        try { data = decodeBundled(); datasetName = "MNIST sample · 6,000 images"; setDatasetLabel(datasetName); sCount.b.textContent = "5,000"; reset(); }
        catch (e) { status.textContent = e.message; }
      }
      run.start();
    },
    stop() { run.stop(); }
  };
});
