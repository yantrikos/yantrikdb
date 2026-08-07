#!/usr/bin/env python3
"""Capability verbs: mount and unmount compiled packs, same as knowledge.

    python packs/cap.py status
    python packs/cap.py mount motion-craft
    python packs/cap.py ask "Build the motion for a login form..."
    python packs/cap.py unmount

`mount` loads the pack's adapter into the resident model (~2s the first
time, instant after) and routes every model="current" request through
it. `unmount` restores the exact base weights — the adapter never
modified them, it sat beside them. The serving daemon is
serve_compiled.py; start it once with --base Qwen/Qwen3.5-4B.
"""
from __future__ import annotations

import json
import sys
import urllib.request

HOST = "http://127.0.0.1:11555"


def call(path: str, payload: dict | None = None, timeout: int = 1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    cmd = args[0]
    if cmd == "status":
        s = call("/api/caps")
        print(f"mounted:    {s.get('mounted') or '(base — nothing mounted)'}")
        print(f"loaded:     {', '.join(s.get('registered', [])) or '-'}")
        print(f"available:  {', '.join(sorted(set(s.get('available', []))))}")
    elif cmd == "mount":
        if len(args) < 2:
            print("mount <pack>", file=sys.stderr)
            return 2
        print(json.dumps(call("/api/caps/mount", {"pack": args[1]})))
    elif cmd == "unmount":
        print(json.dumps(call("/api/caps/unmount", {})))
    elif cmd == "ask":
        prompt = " ".join(args[1:])
        r = call("/api/chat", {
            "model": "current",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": 4000, "temperature": 0.0}})
        print(r.get("message", {}).get("content", ""))
    else:
        print(f"unknown command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
