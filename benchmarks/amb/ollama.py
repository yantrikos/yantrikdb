"""Ollama LLM provider — local models and Ollama Cloud, via the native API.

Why this exists rather than pointing OpenAILLM at Ollama's /v1 endpoint:
Ollama's cloud routing DROPS structured-output enforcement. Both
`response_format: json_schema` (OpenAI-compat) and `format: <schema>`
(native) are accepted and then ignored for `*-cloud` models — verified on
gpt-oss:120b-cloud and deepseek-v4-flash:cloud, which return bare prose.
OpenAILLM's `json.loads` then fails on every call.

The models comply perfectly well when the PROMPT asks for JSON — the same
deepseek-v4-flash that returns prose for BEAM's RAG prompt (which ends
"ANSWER:") returns clean JSON for BEAM's judge prompts (which end
'Respond as JSON: {...}'). So this provider appends an explicit
schema instruction, which is the faithful analogue of what the
schema-enforcing providers already put in front of the model, and parses
tolerantly.

FALLBACK POLICY — deliberately asymmetric, because the two roles have
different failure costs:
  * Every required field is a STRING (the answer path): unparseable output
    is wrapped as the primary string field. Faithful — for a RAG answer the
    prose IS the answer, so nothing is invented and nothing is lost.
  * ANY required field is a boolean or number (the judge paths — `correct`,
    `score`): RAISE. A silent default here would silently corrupt the
    measurement, marking answers right or wrong without the model having
    said so. Loud failure is the only safe behaviour for a scorer.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from .base import LLM, Schema

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 3
_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))

_JSON_TYPES = {"boolean", "integer", "number"}


def _host() -> str:
    """Resolve the Ollama endpoint to something CONNECTABLE.

    `OLLAMA_HOST` is commonly set to a BIND address for the server (this
    machine has `0.0.0.0:11434`). Dialling 0.0.0.0 fails with WinError
    10049 / EADDRNOTAVAIL, so wildcard binds are rewritten to loopback.
    `OMB_OLLAMA_URL` overrides everything for remote endpoints.
    """
    h = os.environ.get("OMB_OLLAMA_URL") or os.environ.get(
        "OLLAMA_HOST", "http://127.0.0.1:11434"
    )
    if not h.startswith("http"):
        h = f"http://{h}"
    for wildcard in ("//0.0.0.0", "//[::]", "//::"):
        if wildcard in h:
            h = h.replace(wildcard, "//127.0.0.1")
            break
    return h.rstrip("/")


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON out of a model response (bare, fenced, or embedded)."""
    text = text.strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        try:
            v = json.loads(fenced.group(1))
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            pass
    # First balanced object in the text.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class OllamaLLM(LLM):
    def __init__(self, model: str = "gpt-oss-backup:20b"):
        self._model = model

    @property
    def model_id(self) -> str:
        return f"ollama:{self._model}"

    def _instruction(self, schema: Schema) -> str:
        fields = []
        for name in schema.required:
            spec = schema.properties.get(name, {})
            t = spec.get("type", "string")
            desc = spec.get("description", "")
            fields.append(f'  "{name}": <{t}>' + (f"  // {desc}" if desc else ""))
        return (
            "\n\nReturn ONLY a single JSON object, no prose before or after, "
            "no markdown fences, with exactly these keys:\n{\n"
            + ",\n".join(fields)
            + "\n}"
        )

    def generate(self, prompt: str, schema: Schema) -> dict:
        body = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": 0.0},
            # Sent even though cloud routing ignores it: local models DO
            # honour it, and it costs nothing where it is dropped.
            "format": {
                "type": "object",
                "properties": schema.properties,
                "required": schema.required,
            },
            "messages": [{"role": "user", "content": prompt + self._instruction(schema)}],
        }
        data = json.dumps(body).encode()
        typed = [
            k for k in schema.required
            if schema.properties.get(k, {}).get("type") in _JSON_TYPES
        ]
        delay = _RETRY_BASE_DELAY
        last_exc = None
        content = ""
        parsed = None
        # Retries cover BOTH transport failures and unusable CONTENT. The
        # cloud endpoint occasionally returns an empty body; observed once in
        # 695 rubric items. That matters more than the rate suggests, because
        # BEAM's `_rubric_item_score` catches every exception and returns 0.0
        # — so a judge hiccup silently becomes a ZERO and biases the measured
        # score DOWN. Asking again is not guessing a verdict, so it is safe
        # here in a way that a default value never is.
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    f"{_host()}/api/chat", data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                    resp = json.load(r)
                content = resp["message"]["content"]
                parsed = _extract_json(content)
                if parsed is not None and all(k in parsed for k in schema.required):
                    return parsed
                # Unusable content. Retry only when we cannot fall back —
                # i.e. when a typed verdict field is required, or nothing at
                # all came back. Prose for an all-string schema is fine.
                if typed or not content.strip():
                    last_exc = ValueError(f"unusable response: {content[:120]!r}")
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(
                    f"Ollama request failed after {_MAX_RETRIES} retries: {last_exc}"
                ) from last_exc

        # See the module docstring: prose is an acceptable answer, never an
        # acceptable score.
        if typed:
            raise ValueError(
                f"{self.model_id} returned no usable JSON for a schema with "
                f"typed field(s) {typed} after {_MAX_RETRIES} attempts; "
                f"refusing to guess a verdict. Last response began: "
                f"{content[:200]!r}"
            )
        out = {k: "" for k in schema.required}
        primary = "answer" if "answer" in schema.required else schema.required[-1]
        out[primary] = content.strip()
        if parsed:
            out.update({k: v for k, v in parsed.items() if k in out})
        return out
