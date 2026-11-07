import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


def rejoin(segments):
    return "".join(s.prefix + s.body + s.suffix for s in segments)


@pytest.mark.parametrize(
    "text",
    [
        "短い文。",
        "  leading and trailing\n\n",
        "段落1。\n\n段落2。\n\n\n\n段落3。\n",
        "あ" * 5000,
        ("これは文です。" * 800 + "\n\n") * 3,
        ("This is a sentence. " * 400 + "\n") * 3,
        "\n\n" + "x" * 2000 + "\n\n" + "y" * 2000 + "\n\n",
    ],
)
def test_split_preserves_text_and_respects_limit(text):
    segments = app.split_text(text)
    assert rejoin(segments) == text
    assert all(s.body and s.body == s.body.strip() for s in segments)
    assert all(len(s.body) <= app.CHUNK_CHARS for s in segments)


def test_short_text_is_one_segment():
    assert app.split_text("段落1。\n\n段落2。") == [app.Segment("", "段落1。\n\n段落2。", "")]


@pytest.mark.parametrize(
    "text,source,target,expected",
    [
        ("会議は月曜日です。", "auto", "auto", "en"),
        ("会議", "auto", "auto", "en"),
        ("Hello world", "auto", "auto", "ja"),
        ("Bonjour", "fr", "auto", "ja"),
        ("こんにちは", "ja", "auto", "en"),
        ("Hello", "auto", "de", "de"),
        ("今天的天气非常好，我们一起去公园散步吧。", "auto", "auto", "ja"),
    ],
)
def test_resolve_target(text, source, target, expected):
    assert app.resolve_target(text, source, target) == expected


def test_parse_builds_template():
    parsed = app.parse_request({
        "text": "This system is convenient.",
        "source_lang": "en",
        "target_lang": "ja",
        "form": "plain",
        "context": " help page ",
        "glossary": [{"source": "system", "target": "仕組み"}, {"source": " ", "target": "x"}],
    })
    assert parsed["template"] == {
        "target_lang": "ja",
        "source_lang": "en",
        "form": "plain",
        "context": "help page",
        "glossary": [{"source": "system", "target": "仕組み"}],
    }


def test_form_is_dropped_for_non_japanese_output():
    parsed = app.parse_request({"text": "こんにちは", "target_lang": "en", "form": "polite"})
    assert "form" not in parsed["template"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"text": "   "},
        {"text": "x", "target_lang": "xx"},
        {"text": "x", "source_lang": "en", "target_lang": "en"},
        {"text": "x <|plamo:tag|> y"},
        {"text": "x", "context": "<|plamo:msg|>"},
        {"text": "x", "form": "casual"},
        {"text": "x", "glossary": ["a=b"]},
    ],
)
def test_rejects_bad_requests(payload):
    with pytest.raises(app.BadRequest):
        app.parse_request(payload)


def test_supported_languages_match_model_card():
    assert len(app.LANGUAGES) == 53


class FakeEngine:
    def __init__(self):
        self.calls = []

    def stream(self, messages, template, max_tokens):
        self.calls.append(messages)
        yield f"T({messages[-1]['content'][:3]})", None
        yield "", "stop"


def test_long_text_streams_with_history_and_keeps_layout():
    text = "\n" + "\n\n".join(["あ" * 1500, "い" * 1500, "う" * 1500, "え" * 1500]) + "\n"
    engine = FakeEngine()
    request = app.parse_request({"text": text, "target_lang": "en"})
    events = list(app.translate_events(engine, request))
    assert events[0]["type"] == "meta" and events[-1]["type"] == "done"
    output = "".join(e["text"] for e in events if e["type"] == "delta")
    assert output == "\nT(あああ)\n\nT(いいい)\n\nT(ううう)\n\nT(えええ)\n"
    assert len(engine.calls[0]) == 1
    assert engine.calls[1][0]["content"].startswith("あ")
    assert engine.calls[1][1] == {"role": "assistant", "content": "T(あああ)"}
    assert len(engine.calls[3]) == 1 + 2 * app.HISTORY_PAIRS  # oldest pair dropped


def test_truncated_output_warns():
    class Truncating(FakeEngine):
        def stream(self, messages, template, max_tokens):
            yield "途中", "length"

    request = app.parse_request({"text": "hello", "target_lang": "ja"})
    assert any(e["type"] == "warning" for e in app.translate_events(Truncating(), request))


def test_download_shows_license_notice_first(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(app, "DEFAULT_MODEL_DIR", tmp_path)
    calls = []

    def fake_download(repo, **kwargs):
        calls.append((repo, kwargs, capsys.readouterr().out))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    app.ensure_model(tmp_path)
    repo, kwargs, printed = calls[0]
    assert repo == app.MODEL_REPO and kwargs["revision"] == app.MODEL_REVISION
    assert "PLaMo Community License" in printed and app.MODEL_LICENSE_URL in printed


def test_no_download_or_notice_when_weights_exist(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {"a": "w.safetensors"}}')
    (tmp_path / "w.safetensors").touch()
    monkeypatch.setattr(app, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("downloaded"))
    app.ensure_model(tmp_path)
    assert capsys.readouterr().out == ""
