"""Classify the ten posts with OpenAI or a local OpenAI-compatible Chat API.

OpenAI (use a Chat Completions-compatible text model):
  Set OPENAI_API_KEY in the environment, then:
  python run_llm.py --base-url https://api.openai.com/v1 --model MODEL_NAME
  If no key is set, an interactive terminal prompts for it without echoing.

Local server (set LOCAL_LLM_API_KEY if authentication is required):
  python run_llm.py --base-url http://localhost:8005/v1 --model MODEL_NAME

Base URLs and full /chat/completions URLs are accepted. Each run is saved
separately under results/rechecks/; the submitted experiment is not overwritten.
Only standard-library modules are needed. Keys are never written to artifacts.
"""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from evaluate import ROOT, load_posts, score_llm


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def api_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or any(c.isspace() for c in value)):
        raise ValueError("Use an HTTP(S) API URL without embedded credentials or whitespace")
    if parsed.scheme == "http" and not is_loopback(parsed.hostname):
        raise ValueError("Remote API connections must use HTTPS")
    parsed.port  # Validate the port before making any request.
    if parsed.query or parsed.fragment:
        raise ValueError("URL query/fragment is not allowed")
    base = value.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[:-len("/chat/completions")]
    elif not parsed.path or parsed.path == "/":
        base += "/v1"
    return base


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a key to a destination other than the URL supplied.
        return None


def request_json(url, key, payload=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    with build_opener(NoRedirects()).open(Request(url, data=body, headers=headers), timeout=timeout) as response:
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


def build_payload(model, text, style):
    payload = {"model": model, "messages": [{"role": "user", "content": text}], "stream": False}
    if style == "compatible":
        # Preserve the settings of the submitted local experiment.
        payload.update(temperature=0, top_p=1, seed=42, max_tokens=4096)
    else:
        # Avoid model-dependent sampling parameters and deprecated max_tokens.
        # Reference: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        payload["max_completion_tokens"] = 8192
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8005/v1", help="API base or full Chat Completions URL")
    parser.add_argument("--model", help="Exact served model ID; otherwise require a single listed model")
    parser.add_argument("--api-key-env", help="Key environment variable (default: OPENAI_API_KEY remotely, LOCAL_LLM_API_KEY locally)")
    parser.add_argument("--api-style", choices=["auto", "openai", "compatible"], default="auto",
                        help="auto: preserve local settings on loopback; OpenAI settings elsewhere")
    parser.add_argument("--out", type=Path, default=ROOT / "results/rechecks", help="Parent directory for separate run folders")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        base = api_url(args.base_url)
        local = is_loopback(urlsplit(base).hostname)
        style = ("compatible" if local else "openai") if args.api_style == "auto" else args.api_style
        key_env = args.api_key_env or ("LOCAL_LLM_API_KEY" if local else "OPENAI_API_KEY")
        key = os.environ.get(key_env)
        if args.api_key_env and not key:
            parser.error(f"Set the {key_env} environment variable (key values must not be saved in code)")
        if not local and not key:
            if not sys.stdin.isatty():
                parser.error(f"Set {key_env}, or run in an interactive terminal to enter the API key")
            key = getpass.getpass("API key (not saved): ")
            if not key.strip():
                parser.error("An API key is required for a remote endpoint")
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        models = None
        model = args.model
        if model is None:
            models, _ = request_json(base + "/models", key, timeout=15)
            ids = [m["id"] for m in models["data"]]
            if len(ids) != 1:
                raise ValueError("Multiple/no served models; specify --model")
            model = ids[0]
        if not model.strip():
            parser.error("--model must not be empty")
        started = datetime.now(timezone.utc)
        folder = args.out.resolve() / started.strftime("%Y%m%dT%H%M%S%fZ")
        folder.mkdir(parents=True, exist_ok=False)
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--out", str(folder)],
                       check=True, capture_output=True)
        llm_input = (folder / "llm_input.txt").read_bytes()
        payload = build_payload(model, llm_input.decode("utf-8"), style)
        save_json(folder / "request.json", payload)
        if models is not None:
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
                "executed_at": started.isoformat(), "interface": "OpenAI-compatible Chat Completions API",
                "api_style": style,
                "base_url": base, "settings": {k: v for k, v in payload.items() if k not in {"model", "messages"}},
                "new_chat": True, "input_sha256": hashlib.sha256(llm_input).hexdigest(),
                "elapsed_seconds": round(elapsed, 3), "usage": response.get("usage"),
                "finish_reason": choice.get("finish_reason"),
                "artifacts": folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else str(folder),
                "server_defaults": "Parameters not present in request use server defaults; not independently verified"}
        save_json(folder / "llm_metadata.json", meta)
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"),
                        "--llm", str(folder / "llm_response.json"),
                        "--metadata", str(folder / "llm_metadata.json"), "--out", str(folder)], check=True)
    except HTTPError as exc:
        # Do not print request headers, credentials, or arbitrary server error bodies.
        hints = {400: "Check model/API style and request parameters", 401: "Check the API key",
                 403: "Check key/project/model access", 404: "Check endpoint URL and model name",
                 429: "Check rate limits, quota and billing"}
        parser.exit(1, f"HTTP {exc.code}: {hints.get(exc.code, 'Check the API server')}. No automatic retry.\n")
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError,
            subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Experiment failed ({type(exc).__name__}); inspect saved artifacts. No automatic retry.\n")


if __name__ == "__main__":
    main()
