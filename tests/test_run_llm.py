from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import run_llm


class LocalRunnerChecks(unittest.TestCase):
    def test_https_remote_and_full_endpoint_are_accepted(self):
        self.assertEqual(run_llm.api_url("https://api.openai.com/v1/chat/completions"),
                         "https://api.openai.com/v1")
        self.assertEqual(run_llm.api_url("https://api.openai.com/"), "https://api.openai.com/v1")
        self.assertEqual(run_llm.api_url("http://[::1]:8005/v1/"), "http://[::1]:8005/v1")

    def test_insecure_urls_and_embedded_secrets_rejected(self):
        for url in ["http://example.com/v1", "http://user:secret@localhost:8005/v1",
                    "http://localhost:8005/v1?api_key=secret", "file:///tmp/key", "https:///v1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                run_llm.api_url(url)

    def test_openai_payload_omits_model_dependent_sampling(self):
        payload = run_llm.build_payload("chosen-model", "input", "openai")
        self.assertEqual(set(payload), {"model", "messages", "stream", "max_completion_tokens"})
        self.assertEqual(payload["max_completion_tokens"], 8192)

    def test_remote_auto_uses_openai_environment_and_no_model_listing(self):
        answer = {"predictions": [{"id": f"{i:02}", "label": "positive"} for i in range(1, 11)]}
        response = {"model": "server-model-snapshot", "choices": [{"finish_reason": "stop",
                    "message": {"content": json.dumps(answer)}}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy2(run_llm.ROOT / "evaluate.py", root / "evaluate.py")
            for name in ("data", "prompts"):
                shutil.copytree(run_llm.ROOT / name, root / name)
            argv = ["run_llm.py", "--base-url", "https://api.openai.com/v1", "--model", "requested-model",
                    "--out", str(root / "checks")]
            with patch.object(run_llm, "ROOT", root), patch.object(sys, "argv", argv), \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "mock-openai-key"}), \
                    patch.object(run_llm, "request_json", return_value=(response, json.dumps(response).encode())) as request:
                run_llm.main()
            request.assert_called_once()
            url, key, payload, timeout = request.call_args.args
            self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
            self.assertEqual(key, "mock-openai-key")
            self.assertEqual(payload["model"], "requested-model")
            self.assertIn("max_completion_tokens", payload)
            self.assertNotIn("seed", payload)
            meta_path = next((root / "checks").glob("*/llm_metadata.json"))
            meta = json.loads(meta_path.read_text("utf-8"))
            self.assertEqual(meta["api_style"], "openai")
            self.assertEqual(meta["response_model"], "server-model-snapshot")
            self.assertNotIn("mock-openai-key", meta_path.read_text("utf-8"))

    def test_missing_remote_key_fails_before_network_noninteractively(self):
        with patch.object(sys, "argv", ["run_llm.py", "--base-url", "https://api.openai.com/v1", "--model", "chosen"]), \
                patch.dict(os.environ, {}, clear=True), patch.object(sys.stdin, "isatty", return_value=False), \
                patch.object(run_llm, "request_json") as request:
            with self.assertRaises(SystemExit) as stopped:
                run_llm.main()
            self.assertEqual(stopped.exception.code, 2)
            request.assert_not_called()

    def test_code_fence_only_is_removed(self):
        self.assertEqual(run_llm.parse_content('```json\n{"predictions": []}\n```'),
                         {"predictions": []})
        with self.assertRaises(ValueError):
            run_llm.parse_content('Explanation: {"predictions": []}')

    def test_http_to_saved_evaluation_without_gold_leak(self):
        self.exercise_http("compatible")

    def test_openai_style_auth_and_model_without_listing(self):
        self.exercise_http("openai")

    def test_auth_failure_does_not_retry_or_expose_key(self):
        self.exercise_http("openai", error=401)

    def test_redirect_does_not_forward_key(self):
        self.exercise_http("openai", error=302)

    def test_truncated_completion_is_not_scored(self):
        self.exercise_http("openai", finish_reason="length")

    def exercise_http(self, style, error=None, finish_reason="stop"):
        captured = []
        get_paths = []
        fixture_key = "fake-key-only-for-unit-test"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode("utf-8"))

            def do_GET(self):
                get_paths.append(self.path)
                self.reply({"data": [{"id": "mock-only-not-an-experiment"}]})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    raise AssertionError("Unexpected endpoint or followed redirect")
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if self.headers.get("Authorization") != "Bearer " + fixture_key:
                    self.send_error(401)
                    return
                if error:
                    self.send_response(error)
                    if error == 302:
                        self.send_header("Location", "/must-not-follow")
                    self.end_headers()
                    self.wfile.write(fixture_key.encode())
                    return
                self.reply({"model": "mock-only-not-an-experiment", "choices": [{
                    "finish_reason": finish_reason, "message": {"content": json.dumps({"predictions": [
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
                (root / "results").mkdir()
                (root / "results/summary.json").write_text("submitted-results-sentinel", encoding="utf-8")
                command = [sys.executable, str(root / "run_llm.py"), "--base-url",
                           f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                           "--api-key-env", "SENTIMENT_TEST_KEY", "--api-style", style]
                if style == "openai":
                    command += ["--model", "explicit-test-model"]
                result = subprocess.run(command, env={**os.environ, "SENTIMENT_TEST_KEY": fixture_key},
                                        capture_output=True, timeout=30)
                succeeded = error is None and finish_reason == "stop"
                self.assertEqual(result.returncode == 0, succeeded, result.stderr.decode())
                self.assertNotIn(fixture_key.encode(), result.stdout + result.stderr)
                self.assertEqual((root / "results/summary.json").read_text("utf-8"), "submitted-results-sentinel")
                runs = list((root / "results/rechecks").iterdir())
                self.assertEqual(len(runs), 1)
                if succeeded:
                    summary = json.loads((runs[0] / "summary.json").read_text("utf-8"))
                    self.assertEqual(summary["llm"]["correct"], 5)
                    self.assertEqual(summary["llm"]["status"], "completed")
                    meta = summary["llm_execution"]
                    self.assertEqual(meta["api_style"], style)
                    self.assertNotIn("quantization", meta)
                    self.assertNotIn("server", meta)
                    self.assertTrue((runs[0] / "api_response.json").exists())
                else:
                    self.assertFalse((runs[0] / "llm_metadata.json").exists())
                    self.assertEqual(json.loads((runs[0] / "summary.json").read_text("utf-8"))["llm"]["status"], "not_run")
                for artifact in runs[0].rglob("*"):
                    if artifact.is_file():
                        self.assertNotIn(fixture_key.encode(), artifact.read_bytes())
            self.assertEqual(len(captured), 1)
            self.assertEqual(get_paths, ["/v1/models"] if style == "compatible" else [])
            if style == "openai":
                self.assertEqual(set(captured[0]), {"model", "messages", "stream", "max_completion_tokens"})
                self.assertEqual(captured[0]["model"], "explicit-test-model")
            else:
                self.assertEqual(captured[0]["temperature"], 0)
                self.assertEqual(captured[0]["seed"], 42)
            self.assertEqual(len(captured[0]["messages"]), 1)
            content = captured[0]["messages"][0]["content"]
            posts = json.loads(content.split("以下に分類対象の投稿をJSONで示します。\n", 1)[1])
            self.assertEqual(len(posts), 10)
            self.assertTrue(all(set(p) == {"id", "text"} for p in posts))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
