"""Run one fixed sentiment experiment against a local OpenAI-compatible server."""
import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from evaluate import ROOT, load_posts, score_llm


def local_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Use a local HTTP(S) URL without embedded credentials")
    host = parsed.hostname
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("Only loopback servers are supported")
        except (ValueError, TypeError):
            raise ValueError("Only localhost/loopback servers are supported") from None
    if parsed.query or parsed.fragment:
        raise ValueError("URL query/fragment is not allowed")
    return value.rstrip("/")


def request_json(url, key, payload=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")), raw


def parse_content(content):
    if not isinstance(content, str):
        raise ValueError("The response has no text content; inspect the saved API response")
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    return json.loads(text)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8005/v1")
    parser.add_argument("--model", help="Exact served model ID; otherwise require a single listed model")
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        base = local_url(args.base_url)
        key = os.environ.get(args.api_key_env)
        models, _ = request_json(base + "/models", key, timeout=15)
        ids = [m["id"] for m in models["data"]]
        model = args.model
        if model is None:
            if len(ids) != 1:
                raise ValueError("Multiple/no served models; specify --model")
            model = ids[0]
        if model not in ids:
            raise ValueError("Requested model is not listed by this server")
        started = datetime.now(timezone.utc)
        folder = ROOT / "results/runs" / started.strftime("%Y%m%dT%H%M%S%fZ")
        folder.mkdir(parents=True, exist_ok=False)
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--out", str(folder)],
                       check=True, capture_output=True)
        llm_input = (folder / "llm_input.txt").read_bytes()
        payload = {"model": model, "messages": [{"role": "user", "content": llm_input.decode("utf-8")}],
                   "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 4096, "stream": False}
        save_json(folder / "request.json", payload)
        save_json(folder / "models.json", models)
        print(f"Running {model}; artifacts: {folder}", flush=True)
        tick = time.monotonic()
        response, raw = request_json(base + "/chat/completions", key, payload, args.timeout)
        elapsed = time.monotonic() - tick
        (folder / "api_response.json").write_bytes(raw)
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Generation did not finish normally; inspect saved response")
        content = choice["message"]["content"]
        answer = parse_content(content)
        score_llm(answer, {p["id"] for p in load_posts(ROOT / "data/posts.csv")})
        save_json(folder / "llm_response.json", answer)
        meta = {"model": model, "response_model": response.get("model"),
                "executed_at": started.isoformat(), "interface": "local OpenAI-compatible Chat Completions API",
                "base_url": base, "settings": {k: v for k, v in payload.items() if k not in {"model", "messages"}},
                "new_chat": True, "input_sha256": hashlib.sha256(llm_input).hexdigest(),
                "quantization": "FP8 (user-reported, downloaded from Hugging Face; exact repository/revision not recorded)",
                "server": "vLLM (user-reported; version not recorded)",
                "elapsed_seconds": round(elapsed, 3), "usage": response.get("usage"),
                "finish_reason": choice.get("finish_reason"), "artifacts": folder.relative_to(ROOT).as_posix(),
                "server_defaults": "Parameters not present in request use server defaults; not independently verified"}
        save_json(folder / "llm_metadata.json", meta)
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"),
                        "--llm", str(folder / "llm_response.json"),
                        "--metadata", str(folder / "llm_metadata.json")], check=True)
        for name in ("summary.json", "predictions.csv"):
            shutil.copy2(ROOT / "results" / name, folder / name)
    except HTTPError as exc:
        # Do not print request headers, credentials, or arbitrary server error bodies.
        parser.exit(1, f"HTTP {exc.code}: check server/authentication. No automatic retry.\n")
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError,
            subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Experiment failed ({type(exc).__name__}); inspect saved artifacts. No automatic retry.\n")


if __name__ == "__main__":
    main()
