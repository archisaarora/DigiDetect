/* DigiDetect: the page. The neural network itself (training, evaluation, prediction) runs in Python;
   this file builds the UI and talks to digidetect/server.py through a small JSON API. */
Site.registerDemo("digidetect", function (root) {
  const { h, fit, loop, colors } = Site;
  const BATCH = 32;
  let state = null, epoch = 0, gen = 0, busy = false;
  let training = false, hidden = 64, epochs = 4;
  let ready = false;
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
          h("span.sc-live", null, h("i"), " PYTHON")),
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
    status.textContent = training ? `Training ${datasetName.toLowerCase()} in Python` : epoch >= epochs ? "Training finished — draw a digit or change the experiment settings." : "Training paused";
  }

  /* ---- talking to the Python server ---- */
  async function api(path, body) {
    const res = await fetch(path, body === undefined ? { cache: "no-store" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
    return json;
  }
  const serverDown = (err) => {
    training = false; ready = false;
    status.textContent = `Can't reach the DigiDetect Python server (${err.message}). Start it with: python -m digidetect`;
  };

  async function loadFullDataset() {
    fullBtn.disabled = true; quickBtn.disabled = true;
    status.textContent = "Connecting to a public MNIST mirror…";
    setDatasetLabel("Loading full MNIST");
    try {
      let s = await api("/api/load_full", {});
      while (s.loading) {
        status.textContent = s.message || "Trying public MNIST sources…";
        await new Promise((r) => setTimeout(r, 400));
        s = await api("/api/status");
      }
      if (s.error) throw new Error(s.error);
      datasetName = "Full MNIST · 70,000 images"; fullDatasetLoaded = true;
      setDatasetLabel(datasetName);
      status.textContent = "Full MNIST is ready. Starting a fresh model…";
      await reset();
    } catch (err) {
      status.textContent = `Full MNIST could not be loaded: ${err.message} The bundled 6,000-image sample is still available.`;
      setDatasetLabel(fullDatasetLoaded ? datasetName : "Bundled sample · offline ready");
    } finally { fullBtn.disabled = false; quickBtn.disabled = false; }
  }

  function loadQuickDataset() {
    datasetName = "MNIST sample · 6,000 images"; fullDatasetLoaded = false;
    setDatasetLabel(datasetName); reset();
  }

  async function reset() {
    const mine = ++gen;
    ready = false;
    try {
      const s = await api("/api/reset", { hidden, epochs, dataset: fullDatasetLoaded ? "full" : "sample" });
      if (mine !== gen) return;
      state = s; ready = true; epoch = s.epoch;
      sCount.b.textContent = s.n_train.toLocaleString("en-US");
      renderStats(); setTraining(true); classify();
    } catch (err) { serverDown(err); }
  }

  function renderStats() {
    const { acc_hist, loss_avg } = state, acc = acc_hist.length ? acc_hist[acc_hist.length - 1] : 0;
    sAcc.b.textContent = `${(acc * 100).toFixed(1)}%${state.done ? " · full test" : ""}`;
    sEpoch.b.textContent = `${state.epoch} / ${state.epochs}`;
    sLoss.b.textContent = (loss_avg || 2.3).toFixed(3);
  }

  /* Ask Python to run a few mini-batches. Requests never overlap. */
  async function trainSome() {
    if (busy) return;
    busy = true; const mine = gen;
    try {
      const s = await api("/api/step", { n: 8 });
      if (mine !== gen) return;
      state = s; epoch = s.epoch; renderStats();
      if (s.done) setTraining(false);
    } catch (err) { serverDown(err); }
    finally { busy = false; }
  }

  function drawChart() {
    const { ctx, w, h: hh } = fit(chart); ctx.clearRect(0, 0, w, hh);
    const accHist = state.acc_hist, lossHist = state.loss_hist;
    const total = Math.max(12, Math.ceil((state.n_train / BATCH) * epochs / 100) + 1);
    const X = (i) => 38 + (i / total) * (w - 52), Y = (v) => hh - 22 - v * (hh - 38);
    ctx.font = "11px ui-monospace, monospace"; ctx.fillStyle = "#718096"; ctx.strokeStyle = "#e6eaf0"; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 0.9, 1]) { ctx.beginPath(); ctx.moveTo(38, Y(v)); ctx.lineTo(w - 10, Y(v)); ctx.stroke(); ctx.fillText(`${Math.round(v * 100)}%`, 3, Y(v) + 4); }
    const line = (arr, color, scale) => { if (!arr.length) return; ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath(); arr.forEach((v, i) => { const y = Y(Math.min(1, v * scale)); i ? ctx.lineTo(X(i), y) : ctx.moveTo(X(i), y); }); ctx.stroke(); };
    line(lossHist, "#a0aaa3", 1 / 2.4); line(accHist, "#477b5b", 1);
    ctx.fillStyle = "#477b5b"; ctx.fillText("accuracy", Math.max(44, w - 84), 14); ctx.fillStyle = "#7a857d"; ctx.fillText("loss", Math.max(44, w - 42), 14);
  }

  function drawTiles() {
    const { ctx, w, h: hh } = fit(tiles); ctx.clearRect(0, 0, w, hh);
    const W = state.tiles;
    const cols = 12, rows = 2, side = 14, cell = Math.floor(Math.min((w - (cols - 1) * 5) / cols, (hh - 5) / rows)), px = cell / side;
    for (let u = 0; u < cols * rows && u < W.length; u++) {
      const x0 = (u % cols) * (cell + 5), y0 = Math.floor(u / cols) * (cell + 5), w1 = W[u];
      let m = 1e-6; for (let i = 0; i < 196; i++) m = Math.max(m, Math.abs(w1[i]));
      for (let i = 0; i < 196; i++) {
        const v = w1[i] / m;
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
  async function drawExample() {
    if (!ready) return;
    let ex; try { ex = await api("/api/example"); } catch (err) { return serverDown(err); }
    const x = ex.x;
    const image = pctx.createImageData(224, 224);
    for (let y = 0; y < 224; y++) for (let xx = 0; xx < 224; xx++) {
      const v = Math.round(x[Math.floor(y / 16) * 14 + Math.floor(xx / 16)] * 255), off = (y * 224 + xx) * 4;
      image.data[off] = v; image.data[off + 1] = v; image.data[off + 2] = v; image.data[off + 3] = 255;
    }
    pctx.putImageData(image, 0, 0); hasInk = true; classify(` · actual label: ${ex.y}`);
  }
  const pos = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * 224 / r.width, (e.clientY - r.top) * 224 / r.height]; };
  pad.addEventListener("pointerdown", (e) => { drawing = true; last = pos(e); pad.setPointerCapture(e.pointerId); stroke(last); });
  pad.addEventListener("pointermove", (e) => { if (drawing) stroke(pos(e)); });
  const end = () => { drawing = false; }; pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
  function stroke(p) {
    pctx.strokeStyle = "#f7f8fa"; pctx.lineWidth = 17; pctx.lineCap = pctx.lineJoin = "round";
    pctx.beginPath(); pctx.moveTo(last[0], last[1]); pctx.lineTo(p[0] + 0.01, p[1]); pctx.stroke(); last = p; hasInk = true; classify();
  }
  function classify(suffix = "") {
    if (!hasInk || !ready) return;
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
    requestPrediction(x, suffix);
  }

  /* Prediction happens in Python. If one is in flight, keep only the newest request. */
  let predBusy = false, predQueued = null;
  async function requestPrediction(x, suffix) {
    if (predBusy) { predQueued = [x, suffix]; return; }
    predBusy = true;
    try {
      const { p } = await api("/api/predict", { x: Array.from(x) });
      let best = 0;
      for (let k = 1; k < 10; k++) if (p[k] > p[best]) best = k;
      barEls.forEach((b, k) => { b.fill.style.width = `${(p[k] * 100).toFixed(1)}%`; b.row.classList.toggle("top", k === best); });
      verdict.textContent = `Prediction: ${best} · ${(p[best] * 100).toFixed(1)}% model confidence${suffix}`;
    } catch (err) { serverDown(err); }
    finally {
      predBusy = false;
      if (predQueued) { const [qx, qs] = predQueued; predQueued = null; requestPrediction(qx, qs); }
    }
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
    start() { if (!state) reset(); run.start(); },
    stop() { run.stop(); }
  };
});
