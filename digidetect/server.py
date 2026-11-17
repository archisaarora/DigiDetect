"""Serves the DigiDetect website and runs the neural network in Python behind it.

The page (index.html, css/, js/) is the original UI. Everything that was machine
learning in the browser (training, evaluating, predicting) now happens here, through a
tiny JSON API. Standard library only for the server; numpy for the maths.
"""
from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import data
from .trainer import Trainer

ROOT = Path(__file__).resolve().parent.parent
TILES = 24                      # hidden neurons shown in the weight grid (12 x 2)


class Session:
    """Holds the current dataset and trainer. One session per server process."""

    def __init__(self):
        self.lock = threading.RLock()
        self.sample = data.load_bundled()
        self.full = None
        self.use_full = False
        self.seed = 1
        self.trainer: Trainer | None = None
        self.load = {"loading": False, "message": "", "error": None, "loaded": False}
        self.reset(64, 4, "sample")

    @property
    def datasets(self):
        return self.full if (self.use_full and self.full) else self.sample

    def reset(self, hidden: int, epochs: int, dataset: str):
        with self.lock:
            self.use_full = dataset == "full" and self.full is not None
            train, test = self.datasets
            self.trainer = Trainer(train, test, int(hidden), int(epochs), seed=1000 + self.seed)
            self.seed += 1
            self.trainer.measure()
            return self.state()

    def step(self, n: int):
        with self.lock:
            self.trainer.train_batches(max(1, min(int(n), 64)))
            return self.state()

    def state(self):
        t = self.trainer
        return {
            "epoch": t.epoch, "epochs": t.epochs, "done": t.done, "step": t.step,
            "loss_avg": t.loss_avg, "acc_hist": t.acc_hist, "loss_hist": t.loss_hist,
            "n_train": t.train.n, "n_test": t.test.n, "hidden": t.hidden,
            "tiles": np.round(t.net.W1[:TILES], 4).tolist(),
        }

    def predict(self, x):
        with self.lock:
            x = np.asarray(x, np.float32).reshape(1, -1)
            return self.trainer.net.forward(x)[1][0].tolist()

    def example(self):
        with self.lock:
            test = self.datasets[1]
            i = int(np.random.randint(test.n))
            return {"x": test.x[i].tolist(), "y": int(test.y[i])}

    def start_full_download(self):
        with self.lock:
            if self.load["loading"]:
                return
            self.load = {"loading": True, "message": "Connecting to a public MNIST mirror…", "error": None, "loaded": False}

        def log(m):
            self.load["message"] = m

        def work():
            try:
                self.full = data.load_full_mnist(log=log)
                self.load = {"loading": False, "message": "", "error": None, "loaded": True}
            except Exception as e:  # noqa: BLE001
                self.load = {"loading": False, "message": "", "error": str(e), "loaded": False}
        threading.Thread(target=work, daemon=True).start()


class Handler(SimpleHTTPRequestHandler):
    session: Session

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):          # keep the terminal quiet
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            return self._json(self.session.state())
        if self.path == "/api/example":
            return self._json(self.session.example())
        if self.path == "/api/status":
            return self._json(self.session.load)
        if self.path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if self.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        # only serve the website files, not the Python source or datasets
        p = self.path.split("?")[0]
        if p in ("/", "/index.html") or p.startswith(("/css/", "/js/")):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            s = self.session
            if self.path == "/api/reset":
                return self._json(s.reset(body.get("hidden", 64), body.get("epochs", 4), body.get("dataset", "sample")))
            if self.path == "/api/step":
                return self._json(s.step(body.get("n", 8)))
            if self.path == "/api/predict":
                return self._json({"p": s.predict(body["x"])})
            if self.path == "/api/load_full":
                s.start_full_download()
                return self._json(s.load)
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)


def serve(port: int = 8000, open_browser: bool = True):
    Handler.session = Session()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"DigiDetect is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
