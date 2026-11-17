"""Starts the real HTTP server on a free port and exercises the JSON API the page uses."""
import json
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from digidetect.server import Handler, Session  # noqa: E402


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Handler.session = Session()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    def test_page_and_assets_are_served_but_source_is_not(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"DigiDetect", r.read())
        for ok in ("/css/style.css", "/js/digidetect.js", "/js/core.js"):
            self.assertEqual(urllib.request.urlopen(self.base + ok).status, 200)
        for hidden in ("/digidetect/nn.py", "/data/digits.npz", "/tests/test_server.py"):
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(self.base + hidden)

    def test_train_until_done_then_predict(self):
        s = self.call("/api/reset", {"hidden": 32, "epochs": 2, "dataset": "sample"})
        self.assertEqual((s["epochs"], s["n_train"], len(s["tiles"]), len(s["tiles"][0])), (2, 5000, 24, 196))
        for _ in range(2000):
            s = self.call("/api/step", {"n": 64})
            if s["done"]:
                break
        self.assertTrue(s["done"])
        self.assertGreater(s["acc_hist"][-1], 0.88)
        ex = self.call("/api/example")
        p = self.call("/api/predict", {"x": ex["x"]})["p"]
        self.assertEqual(len(p), 10)
        self.assertAlmostEqual(sum(p), 1.0, places=4)

    def test_bad_requests_get_json_errors(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.call("/api/predict", {"x": [1, 2, 3]})
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
