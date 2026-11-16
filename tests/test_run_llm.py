from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import run_llm


class LocalRunnerChecks(unittest.TestCase):
    def test_external_urls_and_embedded_secrets_rejected(self):
        for url in ["https://example.com/v1", "http://user:secret@localhost:8005/v1",
                    "http://localhost:8005/v1?api_key=secret"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                run_llm.local_url(url)

    def test_code_fence_only_is_removed(self):
        self.assertEqual(run_llm.parse_content('```json\n{"predictions": []}\n```'),
                         {"predictions": []})
        with self.assertRaises(ValueError):
            run_llm.parse_content('Explanation: {"predictions": []}')

    def test_http_to_saved_evaluation_without_gold_leak(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode("utf-8"))

            def do_GET(self):
                self.reply({"data": [{"id": "mock-only-not-an-experiment"}]})

            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.reply({"model": "mock-only-not-an-experiment", "choices": [{
                    "finish_reason": "stop", "message": {"content": json.dumps({"predictions": [
                        {"id": f"{i:02}", "label": "positive"} for i in range(10, 0, -1)]})}}]})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for name in ["evaluate.py", "run_llm.py"]:
                    shutil.copy2(run_llm.ROOT / name, root / name)
                for name in ["data", "prompts"]:
                    shutil.copytree(run_llm.ROOT / name, root / name)
                result = subprocess.run([sys.executable, str(root / "run_llm.py"), "--base-url",
                                         f"http://127.0.0.1:{server.server_port}/v1"],
                                        capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                summary = json.loads((root / "results/summary.json").read_text("utf-8"))
                self.assertEqual(summary["llm"]["correct"], 5)
                self.assertEqual(summary["llm"]["status"], "completed")
                runs = list((root / "results/runs").iterdir())
                self.assertEqual(len(runs), 1)
                self.assertTrue((runs[0] / "api_response.json").exists())
            self.assertEqual(len(captured), 1)
            self.assertEqual(len(captured[0]["messages"]), 1)
            content = captured[0]["messages"][0]["content"]
            posts = json.loads(content.split("以下に分類対象の投稿をJSONで示します。\n", 1)[1])
            self.assertEqual(len(posts), 10)
            self.assertTrue(all(set(p) == {"id", "text"} for p in posts))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
