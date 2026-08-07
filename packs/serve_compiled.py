#!/usr/bin/env python3
"""Serve base and compiled adapters from one process, as an Ollama endpoint.

evaluate.py talks to `POST /api/chat`. This speaks that, so a compiled
pack is measured by the existing harness with the existing grader and no
changes to either — the model name selects whether the adapter is
mounted:

    base                 adapter disabled
    <pack>               that pack's adapter mounted

Both names are served by ONE set of bf16 weights in ONE process, with
peft toggling the adapter between requests. That is the point. If the
base arm ran through Ollama at Q4 and the compiled arm ran here at bf16,
any difference between them could be the quantisation rather than the
adapter, and the experiment would prove nothing. Here the only
difference between arms is `enable_adapters()` versus
`disable_adapters()` on the same tensors.

It is also the mount/unmount demonstration: swapping capability is a
method call on a resident model, not a reload.

    .venv-compile/Scripts/python packs/serve_compiled.py \
        --adapter mcp-spec --adapter c-safety

    python packs/evaluate.py --host http://127.0.0.1:11555 \
        --model base --model mcp-spec --pack mcp-spec
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "training"

from compile import chat_prefix  # noqa: E402  - one prompt builder, both sides

STATE: dict = {}
LOCK = threading.Lock()


def resolve_cap(name: str, tag: str | None, db: str | None):
    """Find a capability, preferring what was INSTALLED over what was built.

    Installed caps are the distributable artifact — verified, digest-bound
    and sitting beside their database. Training runs are a developer
    convenience and are searched second, so a machine that has both
    serves the one a user actually installed.

    A cap whose manifest names a different base revision than the loaded
    model is REFUSED rather than mounted: a LoRA is a delta against
    specific weights, and applied to the wrong ones it is noise wearing
    a capability's name.
    """
    import json as _j

    if db:
        p = Path(db)
        d = p.with_name(p.stem + ".caps") / name
        mf = d / "capability.json"
        if mf.exists():
            man = _j.loads(mf.read_text(encoding="utf-8"))
            want = (man.get("base") or {}).get("revision")
            have = STATE.get("base_revision")
            if want and have and want != have:
                return None, (f"{name} was compiled against base revision "
                              f"{want[:12]}, this server is running {have[:12]} "
                              f"— refusing to mount a delta for different weights")
            return d, ""
    for t in (tag, "v1", "adapter", "v2"):
        if not t:
            continue
        for n in (name, f"{name}-craft"):
            if (RUNS / n / t).exists():
                return RUNS / n / t, ""
    return None, f"no capability named {name} installed or built"


def load(base: str, adapters: list[str], tag: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    loaded = []
    for i, name in enumerate(adapters):
        path = RUNS / name / tag
        if not path.exists():
            print(f"  no adapter at {path} — skipping {name}", file=sys.stderr)
            continue
        if not loaded:
            model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
        else:
            model.load_adapter(str(path), adapter_name=name)
        loaded.append(name)
        meta = path / "compile.json"
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        print(f"  mounted {name}  ({info.get('adapter_bytes', 0) / 1e6:.1f} MB, "
              f"{info.get('examples', '?')} examples, "
              f"{info.get('steps', '?')} steps)")

    ref = Path.home() / ".cache" / "huggingface" / "hub" /         f"models--{base.replace('/', '--')}" / "refs" / "main"
    STATE.update(tok=tok, model=model, adapters=loaded, base=base,
                 base_revision=ref.read_text(encoding="utf-8").strip()
                 if ref.exists() else None,
                 is_peft=bool(loaded))
    return model


def generate(model_name: str, messages: list[dict], num_predict: int,
             temperature: float) -> str:
    import torch

    tok, model = STATE["tok"], STATE["model"]
    with LOCK:
        model = STATE["model"]
        if STATE["is_peft"]:
            # `enable_adapters` / `disable_adapters` are transformers'
            # own PEFT-integration methods and raise "No adapter loaded"
            # on a model wrapped by PeftModel.from_pretrained, because
            # that path never sets `_hf_peft_config_loaded`. The layer
            # toggles on the LoraModel are the ones that apply here, and
            # they fail loudly rather than silently serving the wrong
            # arm.
            lora = model.base_model
            if model_name == "current":
                # `current` reads GLOBAL mount state, so it is only
                # meaningful to ONE client at a time. Two scripts sharing
                # this daemon produced byte-identical "compiled" and
                # "bare" artifacts once — each was toggling a flag the
                # other was reading, and nothing in the output looked
                # wrong. Measurement code must name its adapter
                # explicitly; `current` is for interactive use.
                if STATE.get("clients", 0) > 1:
                    raise RuntimeError(
                        "model='current' is unsafe while more than one client "
                        "is using this daemon: mount state is global. Name the "
                        "adapter (or 'base') explicitly instead.")
                if STATE.get("mounted"):
                    model.set_adapter(STATE["mounted"])
                    lora.enable_adapter_layers()
                else:
                    lora.disable_adapter_layers()
            elif model_name in STATE["adapters"]:
                model.set_adapter(model_name)
                lora.enable_adapter_layers()
            else:
                # Every name that is not a loaded adapter is the base
                # model. Naming it explicitly rather than defaulting
                # silently keeps a typo from being scored as an arm.
                lora.disable_adapter_layers()

        # The SAME function training used. See compile.chat_prefix for
        # what happens when these two drift.
        ids = tok(chat_prefix(tok, messages), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=num_predict,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=None if temperature <= 0 else 0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = out[0][ids["input_ids"].shape[1]:]
        answer = tok.decode(gen, skip_special_tokens=True)

    # Qwen emits a reasoning block even with thinking suppressed in some
    # template versions. The harness asks for two sentences and grades
    # the text it gets, so an unclosed think block would be scored as the
    # answer.
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    return answer.strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: A003 - quiet
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            names = ["base"] + STATE.get("adapters", [])
            self._send(200, {"models": [{"name": n} for n in names]})
        elif self.path.startswith("/api/caps"):
            installed = []
            if STATE.get("db"):
                dbp = Path(STATE["db"])
                cd = dbp.with_name(dbp.stem + ".caps")
                installed = sorted(x.name for x in cd.glob("*/capability.json")
                                   for x in [x.parent])
            self._send(200, {"registered": STATE.get("adapters", []),
                             "mounted": STATE.get("mounted"),
                             "installed": installed,
                             "base_revision": STATE.get("base_revision"),
                             "built": sorted({p.parent.parent.name
                                              for p in RUNS.glob("*/*/adapter_config.json")})})
        else:
            self._send(404, {"error": "not found"})

    def _note_client(self):
        """Distinct peers seen this process. Cheap, and it is the signal
        the `current` guard needs to refuse an ambiguous request."""
        seen = STATE.setdefault("client_addrs", set())
        seen.add(self.client_address[0] + ":" + str(self.headers.get("User-Agent", ""))[:40])
        STATE["clients"] = len(seen)

    def do_POST(self):
        self._note_client()
        # SECONDARY INTERFACE — single operator only. Read this before
        # using it for anything that produces a number or serves more
        # than one caller.
        #
        # The primary interface is naming the adapter PER REQUEST:
        #
        #   {"model": "motion-craft-craft", ...}   adapter applied
        #   {"model": "base", ...}                 adapter disabled
        #
        # That is the shape every measured result in this repo uses, and
        # it is the shape vLLM exposes natively (an adapter is a model
        # id). A request that names its adapter cannot be answered by
        # whatever the previous request left mounted.
        #
        # The mount/unmount pair below is a convenience for a human at a
        # terminal. It sets state on the SERVER, which llama.cpp's
        # POST /lora-adapters does too — and that is a production hazard
        # rather than a nicety: with N agents against one serving
        # process, one agent mounting changes every other agent's
        # weights mid-session and nothing in any output looks wrong.
        # This exact failure already cost a measurement here, where two
        # scripts sharing this daemon produced byte-identical "compiled"
        # and "bare" artifacts.
        #
        #   POST /api/caps/mount   {"pack": "motion-craft", "tag": "v1"}
        #   POST /api/caps/unmount {}
        #
        # Loading a new adapter costs ~2s once; the flag flip afterwards
        # is milliseconds.
        if self.path.startswith("/api/caps/"):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "bad json"})
                return
            with LOCK:
                if self.path.startswith("/api/caps/mount"):
                    name = req.get("pack", "")
                    cand, why = resolve_cap(name, req.get("tag"),
                                            req.get("db") or STATE.get("db"))
                    if cand is None:
                        self._send(404, {"error": why})
                        return
                    model = STATE["model"]
                    if name not in STATE["adapters"]:
                        from peft import PeftModel
                        if STATE["is_peft"]:
                            model.load_adapter(str(cand), adapter_name=name)
                        else:
                            STATE["model"] = model = PeftModel.from_pretrained(
                                model, str(cand), adapter_name=name)
                            STATE["is_peft"] = True
                        STATE["adapters"].append(name)
                    model = STATE["model"]
                    model.set_adapter(name)
                    model.base_model.enable_adapter_layers()
                    STATE["mounted"] = name
                    self._send(200, {"mounted": name})
                else:
                    if STATE.get("is_peft"):
                        STATE["model"].base_model.disable_adapter_layers()
                    was, STATE["mounted"] = STATE.get("mounted"), None
                    self._send(200, {"unmounted": was})
            return
        if not self.path.startswith("/api/chat"):
            self._send(404, {"error": "only /api/chat and /api/caps/*"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return
        opts = req.get("options") or {}
        try:
            answer = generate(
                req.get("model", "base"),
                req.get("messages", []),
                int(opts.get("num_predict", 400)),
                float(opts.get("temperature", 0.0)),
            )
        except Exception as e:                                 # noqa: BLE001
            # evaluate.py refuses to report a score when calls fail, and
            # that check only fires if the failure reaches it as text.
            self._send(200, {"message": {"content": f"<<error: {e}>>"}})
            return
        self._send(200, {"message": {"role": "assistant", "content": answer},
                         "done": True})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", action="append", default=[],
                    help="pack name whose adapter to mount; repeatable")
    ap.add_argument("--tag", default="adapter")
    ap.add_argument("--db", help="database whose <stem>.caps/ holds installed capabilities")
    ap.add_argument("--port", type=int, default=11555)
    a = ap.parse_args()

    print(f"loading {a.base}")
    load(a.base, a.adapter, a.tag)
    STATE["db"] = a.db
    names = ["base"] + STATE["adapters"]
    print(f"\n  serving {', '.join(names)} on http://127.0.0.1:{a.port}")
    print("  point evaluate.py at it with --host\n")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
