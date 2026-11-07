#!/usr/bin/env python3
"""Local translation UI for PLaMo 3 Translate on Apple silicon.

Runs the model's bundled mlx-lm server (serve.py) as a child process on a private
port, and serves the web UI plus a small streaming translation API on 127.0.0.1.
Everything stays on this machine: the child runs with Hugging Face offline mode on.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, NamedTuple

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"
LOG_FILE = ROOT / "logs" / "mlx-server.log"

MODEL_REPO = "pfnet/plamo-3-translate-8b-MLX-4bit"
# The model runs bundled code (trust_remote_code), so pin the revision that was reviewed.
MODEL_REVISION = "9d87684408bdf34500b7d6115c257d83350c1fe5"
DEFAULT_MODEL_DIR = ROOT / "models" / "plamo-3-translate-8b-MLX-4bit"
MODEL_LICENSE_URL = "https://plamo.preferredai.jp/info/plamo-community-license"
MODEL_NOTICE = f"""モデルをダウンロードします（約 6 GB）: {MODEL_REPO}

Built with PLaMo. モデルには MIT License ではなく PLaMo Community License が適用され、
利用すると同ライセンスに同意したことになります。商用利用には PFN への登録または個別契約が必要です。
ライセンス本文: {MODEL_LICENSE_URL}
同意しない場合は Ctrl+C で中止してください。
"""

# Long text is translated piece by piece. The model limit is 32,768 tokens for input and output
# combined; these keep a chunk, its history and its output comfortably below that.
CHUNK_CHARS = 2000
HISTORY_PAIRS = 2
MAX_OUTPUT_TOKENS = 8192
MAX_BODY_BYTES = 5 * 1024 * 1024
RESERVED_TOKEN = "<|plamo:"
STOP_TOKENS = ["<|plamo:tag|>", "<|plamo:bos|>"]

LANGUAGES = {
    "ja": "日本語", "en": "英語", "zh-hans": "中国語（簡体字）", "zh-hant": "中国語（繁体字）",
    "ko": "韓国語", "de": "ドイツ語", "fr": "フランス語", "es": "スペイン語", "it": "イタリア語",
    "ru": "ロシア語", "nl": "オランダ語", "pl": "ポーランド語", "vi": "ベトナム語",
    "th": "タイ語", "id": "インドネシア語", "ms": "マレー語", "tl": "タガログ語", "hi": "ヒンディー語",
    "ar": "アラビア語", "az": "アゼルバイジャン語", "be": "ベラルーシ語",
    "bg": "ブルガリア語", "bn": "ベンガル語", "ca": "カタルーニャ語", "ceb": "セブアノ語",
    "et": "エストニア語", "eu": "バスク語", "fa": "ペルシア語", "gu": "グジャラート語",
    "he": "ヘブライ語", "hr": "クロアチア語", "hu": "ハンガリー語", "hy": "アルメニア語",
    "jv": "ジャワ語", "ka": "ジョージア語", "kk": "カザフ語", "km": "クメール語",
    "lt": "リトアニア語", "lv": "ラトビア語", "ml": "マラヤーラム語", "mr": "マラーティー語",
    "my": "ビルマ語", "ne": "ネパール語", "pa": "パンジャブ語", "si": "シンハラ語",
    "sk": "スロバキア語", "sl": "スロベニア語", "sr": "セルビア語", "sw": "スワヒリ語",
    "ta": "タミル語", "te": "テルグ語", "ur": "ウルドゥー語", "uz": "ウズベク語",
}


class BadRequest(ValueError):
    """The request cannot be translated; the message is shown to the user."""


class EngineError(RuntimeError):
    """The inference server failed; the message is shown to the user."""


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------


class Segment(NamedTuple):
    """A piece of text to translate, with the whitespace around it kept verbatim."""

    prefix: str
    body: str
    suffix: str


def _pieces(text: str, limit: int) -> Iterator[str]:
    """Yield pieces of at most `limit` chars that concatenate back to `text`."""
    if len(text) <= limit:
        yield text
        return
    for sentence in re.findall(r".+?(?:[。！？.!?]+[\s」』）)\"']*|\n+|$)", text, flags=re.S):
        for start in range(0, len(sentence), limit):
            yield sentence[start : start + limit]


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[Segment]:
    """Split text at paragraph, then sentence, boundaries so ''.join of all parts equals text."""
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"(?<=\n\n)", text):
        for piece in _pieces(paragraph, limit):
            if current and len(current) + len(piece) > limit:
                chunks.append(current)
                current = ""
            current += piece
    if current:
        chunks.append(current)

    segments: list[Segment] = []
    pending = ""  # whitespace-only chunks attach to the next segment
    for chunk in chunks:
        body = chunk.strip()
        if not body:
            pending += chunk
            continue
        start = chunk.index(body)
        segments.append(Segment(pending + chunk[:start], body, chunk[start + len(body) :]))
        pending = ""
    if pending and segments:
        last = segments[-1]
        segments[-1] = last._replace(suffix=last.suffix + pending)
    return segments


def looks_japanese(text: str) -> bool:
    """Heuristic for the automatic JA<->EN direction."""
    if re.search(r"[぀-ヿｦ-ﾟ]", text):
        return True
    # Short kanji-only text such as 会議 has no kana but is most likely Japanese here.
    has_kanji = re.search(r"[一-鿿]", text) is not None
    return has_kanji and len(text.strip()) <= 12 and not re.search(r"[A-Za-z]", text)


def resolve_target(text: str, source: str, target: str) -> str:
    if target != "auto":
        return target
    if source != "auto":
        return "en" if source == "ja" else "ja"
    return "en" if looks_japanese(text) else "ja"


def max_tokens_for(text: str) -> int:
    return min(MAX_OUTPUT_TOKENS, 3 * len(text) + 256)


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise BadRequest(f"{name}は文字列で指定してください。")
    if RESERVED_TOKEN in value:
        raise BadRequest(f"{name}にモデルの予約語「{RESERVED_TOKEN}」を含めることはできません。")
    return value


def parse_request(payload: Any) -> dict[str, Any]:
    """Validate a translate request and return normalized options."""
    if not isinstance(payload, dict):
        raise BadRequest("リクエストの形式が正しくありません。")
    text = _require_text(payload.get("text", ""), "翻訳するテキスト")
    if not text.strip():
        raise BadRequest("翻訳するテキストを入力してください。")

    source = payload.get("source_lang", "auto")
    target = payload.get("target_lang", "auto")
    if source != "auto" and source not in LANGUAGES:
        raise BadRequest(f"未対応の翻訳元言語です: {source}")
    if target != "auto" and target not in LANGUAGES:
        raise BadRequest(f"未対応の翻訳先言語です: {target}")
    resolved = resolve_target(text, source, target)
    if source == resolved:
        raise BadRequest("翻訳元と翻訳先に同じ言語は指定できません。")

    form = payload.get("form")
    if form not in (None, "", "polite", "plain"):
        raise BadRequest("文体の指定が正しくありません。")

    glossary = []
    for term in payload.get("glossary") or []:
        if not isinstance(term, dict):
            raise BadRequest("用語集の形式が正しくありません。")
        src = _require_text(term.get("source"), "用語集").strip()
        dst = _require_text(term.get("target"), "用語集").strip()
        if src and dst:
            glossary.append({"source": src, "target": dst})

    template: dict[str, Any] = {"target_lang": resolved}
    if source != "auto":
        template["source_lang"] = source
    if form and resolved == "ja":  # the model only supports style selection for Japanese output
        template["form"] = form
    context = _require_text(payload.get("context") or "", "コンテキスト").strip()
    if context:
        template["context"] = context
    if glossary:
        template["glossary"] = glossary
    return {"text": text, "template": template, "target": resolved}


# ---------------------------------------------------------------------------
# Inference engine (bundled mlx-lm server)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_model(model_dir: Path) -> None:
    """Download the default model if its weights are not all present."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if all((model_dir / shard).is_file() for shard in shards):
            return
    if model_dir != DEFAULT_MODEL_DIR:
        sys.exit(f"モデルが見つかりません: {model_dir}")
    from huggingface_hub import snapshot_download

    print(MODEL_NOTICE, flush=True)
    snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        ignore_patterns=["figures/*"],
    )


class Engine:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.port = _free_port()
        self.state = "starting"  # starting -> loading -> ready | error
        self.message = "推論サーバーを起動しています…"
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        LOG_FILE.parent.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        with LOG_FILE.open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, "serve.py", "--model", ".", "--trust-remote-code",
                 "--host", "127.0.0.1", "--port", str(self.port)],
                cwd=self.model_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        threading.Thread(target=self._warm_up, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _watch(self) -> None:
        code = self.process.wait()
        if self.state != "stopped":
            self.state = "error"
            self.message = f"推論サーバーが停止しました（終了コード {code}）。詳細: {LOG_FILE}"

    def _warm_up(self) -> None:
        """Load the weights ahead of the first real request and report readiness."""
        while self.state == "starting":
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                conn.request("GET", "/v1/models")
                conn.getresponse().read()
                conn.close()
                self.state = "loading"
                self.message = "モデルを読み込んでいます…"
            except OSError:
                time.sleep(0.5)
        if self.state != "loading":
            return
        try:
            for _ in self.stream([{"role": "user", "content": "Hello."}], {"target_lang": "ja"}, 8):
                pass
        except (EngineError, OSError) as error:
            if self.state == "loading":
                self.state = "error"
                self.message = f"モデルの読み込みに失敗しました: {error}"
            return
        if self.state == "loading":
            self.state = "ready"
            self.message = "準備完了"

    def stream(self, messages: list[dict], template: dict, max_tokens: int) -> Iterator[tuple[str, str | None]]:
        """Yield (text, finish_reason) from one streaming chat completion."""
        body = json.dumps({
            "model": ".",
            "messages": messages,
            "chat_template_kwargs": template,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stop": STOP_TOKENS,
            "stream": True,
        })
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        try:
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                detail = response.read().decode("utf-8", "replace")[:500]
                raise EngineError(f"推論サーバーのエラー（{response.status}）: {detail}")
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if "error" in chunk:
                    raise EngineError(f"推論サーバーのエラー: {chunk['error']}")
                choice = chunk["choices"][0]
                yield choice.get("delta", {}).get("content") or "", choice.get("finish_reason")
        finally:
            conn.close()

    def is_ready(self) -> bool:
        return self.state == "ready"


def translate_events(engine: Engine, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield UI events for a validated request, translating long text chunk by chunk."""
    segments = split_text(request["text"])
    yield {"type": "meta", "target_lang": request["target"], "chunks": len(segments)}
    history: list[dict[str, str]] = []
    for index, segment in enumerate(segments, start=1):
        yield {"type": "progress", "chunk": index, "chunks": len(segments)}
        if segment.prefix:
            yield {"type": "delta", "text": segment.prefix}
        translated = ""
        finish = None
        messages = [*history, {"role": "user", "content": segment.body}]
        with closing(engine.stream(messages, request["template"], max_tokens_for(segment.body))) as stream:
            for text, reason in stream:
                finish = reason or finish
                if text:
                    translated += text
                    yield {"type": "delta", "text": text}
        if finish == "length":
            yield {"type": "warning", "message": "出力が長さの上限に達したため、訳文が途中で切れている可能性があります。"}
        if segment.suffix:
            yield {"type": "delta", "text": segment.suffix}
        if translated.strip():
            history += [
                {"role": "user", "content": segment.body},
                {"role": "assistant", "content": translated.strip()},
            ]
            history = history[-2 * HISTORY_PAIRS :]
    yield {"type": "done"}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "PLaMoTranslate"
    protocol_version = "HTTP/1.1"
    engine: Engine
    model_name: str

    def log_message(self, *args) -> None:  # keep source text and traffic out of the terminal
        pass

    def _trusted(self) -> bool:
        """Only answer our own page: blocks DNS rebinding and cross-site POSTs to localhost."""
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{host}" for host in hosts}

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(value, ensure_ascii=False).encode())

    def do_GET(self) -> None:
        if not self._trusted():
            self._json(403, {"error": "forbidden"})
        elif self.path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.read_bytes())
        elif self.path == "/api/status":
            self._json(200, {
                "app": "plamo-translate",
                "state": self.engine.state,
                "message": self.engine.message,
                "model": self.model_name,
                "languages": LANGUAGES,
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._trusted():
            self._json(403, {"error": "forbidden"})
            return
        if self.path != "/api/translate":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._json(413, {"error": "テキストが長すぎます。"})
            return
        try:
            request = parse_request(json.loads(self.rfile.read(length)))
        except json.JSONDecodeError:
            self._json(400, {"error": "リクエストの形式が正しくありません。"})
            return
        except BadRequest as error:
            self._json(400, {"error": str(error)})
            return
        if not self.engine.is_ready():
            self._json(503, {"error": self.engine.message})
            return

        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            with closing(translate_events(self.engine, request)) as events:
                try:
                    for event in events:
                        self._event(event)
                except (EngineError, OSError) as error:
                    if isinstance(error, (BrokenPipeError, ConnectionResetError)):
                        raise
                    self._event({"type": "error", "message": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the user pressed stop; closing the generator cancels the upstream request

    def _event(self, event: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()


def _existing_app_is_running(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/api/status", headers={"Host": f"127.0.0.1:{port}"})
        running = json.loads(conn.getresponse().read()).get("app") == "plamo-translate"
        conn.close()
        return running
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="UI port (default: 8765)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, help="local MLX model directory")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/"

    if _existing_app_is_running(args.port):
        print(f"すでに起動しています: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return

    model_dir = args.model.resolve()
    ensure_model(model_dir)

    engine = Engine(model_dir)
    Handler.engine = engine
    Handler.model_name = model_dir.name
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True

    def shutdown(*_):
        engine.state = "stopped"
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGHUP, shutdown)

    engine.start()
    print(f"PLaMo Translate: {url}（Ctrl+C で終了）", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        engine.state = "stopped"
        engine.stop()


if __name__ == "__main__":
    main()
