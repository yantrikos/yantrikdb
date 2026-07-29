#!/usr/bin/env python3
"""Use-case validation: can a small model DO the thing, pack mounted?

Four packs spanning the boundary of what a pack can claim:

  shakespeare-voice   style        (craft-rule compliance)
  einstein-method     method       (procedure compliance)
  react-craft         framework    (house-rule compliance)
  wordpress-expert    domain       (platform + security craft)

Honesty contract, stated up front: a pack shifts KNOWLEDGE and
PROCEDURE, not raw capability. The claim under test is never "as good as
Shakespeare/Einstein/Claude" — it is "measurably more compliant with the
named craft, on checks written down before the run." Every check is a
deterministic function of the output text; no LLM judges anything.

Two conditions per task, same model and prompt: baseline (no pack) and
mounted (pack_context in the system prompt + similarity-gated retrieval
in the user turn). The per-task checks are reported individually, so a
pack that wins on pronouns and loses on meter shows exactly that.

Usage:
    python packs/evaluate_usecases.py --model qwen3.5:4b
    python packs/evaluate_usecases.py --model qwen3.5:4b --case react-craft
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from yantrikdb import YantrikDB

import evaluate
from evaluate import DIST, MIN_SIMILARITY, resolve_host

HERE = Path(__file__).resolve().parent

BASE_SYSTEM = (
    "You are a capable assistant. Complete the task directly. When asked for "
    "code, output the code itself with no surrounding essay."
)

# ── Deterministic graders ────────────────────────────────────────────
# Each check: (name, fn(text) -> bool). A task names which checks apply.
# Written before the first run and not tuned afterwards.

ARCHAIC = re.compile(
    r"\b(thou|thee|thy|thine|hath|doth|art|dost|hast|wherefore|prithee|ere|oft|anon|"
    r"morrow|'tis|'twas|o'er|ne'er|e'en|hither|'twixt)\b",
    re.IGNORECASE,
)
MODERNISM = re.compile(
    r"\b(okay|ok|guys?|cool|awesome|nice|hello|hi|yeah|stuff|really|amazing|"
    r"computer|phone|email|internet|website|app)\b",
    re.IGNORECASE,
)


def syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    n = len(groups)
    if word.lower().endswith("e") and n > 1 and not word.lower().endswith(("le", "ye")):
        n -= 1
    return max(1, n)


def verse_lines(text: str) -> list[str]:
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("#", "*", "-", ">"))
    ]


def check_archaic_density(text: str) -> bool:
    """At least 5 distinct Early Modern markers."""
    return len({m.lower() for m in ARCHAIC.findall(text)}) >= 5


def check_no_modernisms(text: str) -> bool:
    return not MODERNISM.search(text)


def check_verse_shape(text: str) -> bool:
    """Majority of lines within 8-13 estimated syllables — blank-verse
    shaped, not prose. Syllable estimation is rough and pre-registered
    as such; the window is wide to absorb its error."""
    lines = verse_lines(text)
    if len(lines) < 4:
        return False
    ok = sum(1 for ln in lines if 8 <= sum(syllables(w) for w in re.findall(r"[A-Za-z']+", ln)) <= 13)
    return ok / len(lines) >= 0.6


def check_thought_experiment(text: str) -> bool:
    return bool(re.search(r"\b(imagine|suppose|consider|picture)\b", text, re.I))


def check_invariance(text: str) -> bool:
    return bool(
        re.search(
            r"\b(invariant|frame of reference|reference frame|same for (all|both|every)|"
            r"observer-dependent|does not depend on the observer|all observers)\b",
            text,
            re.I,
        )
    )


def check_limit_case(text: str) -> bool:
    return bool(
        re.search(
            r"\b(limit(ing)? case|in the limit|extreme case|approaches (zero|infinity|c\b)|"
            r"tends to (zero|infinity)|as .{1,30}(goes|tends|approaches))\b",
            text,
            re.I,
        )
    )


def check_operational(text: str) -> bool:
    return bool(
        re.search(
            r"\b(operational|how (would|do) (we|you|one) measure|measured by|"
            r"measurement procedure|clock reading|light signal|synchroniz)\b",
            text,
            re.I,
        )
    )


def check_function_component(text: str) -> bool:
    has_fn = bool(re.search(r"(function\s+[A-Z]\w*|const\s+[A-Z]\w*\s*=)", text))
    no_class = not re.search(r"class\s+\w+\s+extends\s+(React\.)?Component", text)
    return has_fn and no_class


def check_map_has_key(text: str) -> bool:
    if ".map(" not in text:
        return False
    return "key=" in text


def check_key_not_index(text: str) -> bool:
    return not re.search(r"key=\{?\s*(index|i|idx)\s*\}?", text)


def check_img_alt(text: str) -> bool:
    imgs = re.findall(r"<img\b[^>]*>", text)
    return bool(imgs) and all("alt=" in i for i in imgs)


def check_labelled_inputs(text: str) -> bool:
    if "<input" not in text:
        return False
    return "htmlFor" in text or "aria-label" in text or "<label" in text


def check_functional_update(text: str) -> bool:
    return bool(re.search(r"set[A-Z]\w*\(\s*\w+\s*=>", text))


def check_no_div_onclick(text: str) -> bool:
    return not re.search(r"<div\b[^>]*onClick", text)


def check_wp_escapes_output(text: str) -> bool:
    return bool(re.search(r"esc_(html|attr|url)|wp_kses", text))


def check_wp_sanitizes_input(text: str) -> bool:
    if not re.search(r"\$_(POST|GET|REQUEST)", text):
        return True  # nothing read, nothing to sanitize
    return bool(re.search(r"sanitize_\w+|absint\s*\(|wp_unslash", text))


def check_wp_prepared_query(text: str) -> bool:
    if "$wpdb" not in text:
        return True
    return "prepare(" in text


def check_wp_nonce(text: str) -> bool:
    return bool(re.search(r"wp_nonce_field|wp_verify_nonce|check_admin_referer|check_ajax_referer", text))


def check_wp_capability(text: str) -> bool:
    return "current_user_can" in text


def check_wp_hooks(text: str) -> bool:
    return bool(re.search(r"add_(action|filter)\s*\(", text))


def check_wp_prefix(text: str) -> bool:
    """Public names carry a plugin prefix rather than bare generic names."""
    fns = re.findall(r"function\s+([a-z_][a-z0-9_]*)\s*\(", text)
    if not fns:
        return True
    return all("_" in f for f in fns)


CHECKS = {
    "archaic_density": check_archaic_density,
    "no_modernisms": check_no_modernisms,
    "verse_shape": check_verse_shape,
    "thought_experiment": check_thought_experiment,
    "invariance": check_invariance,
    "limit_case": check_limit_case,
    "operational": check_operational,
    "function_component": check_function_component,
    "map_has_key": check_map_has_key,
    "key_not_index": check_key_not_index,
    "img_alt": check_img_alt,
    "labelled_inputs": check_labelled_inputs,
    "functional_update": check_functional_update,
    "no_div_onclick": check_no_div_onclick,
    "wp_escapes_output": check_wp_escapes_output,
    "wp_sanitizes_input": check_wp_sanitizes_input,
    "wp_prepared_query": check_wp_prepared_query,
    "wp_nonce": check_wp_nonce,
    "wp_capability": check_wp_capability,
    "wp_hooks": check_wp_hooks,
    "wp_prefix": check_wp_prefix,
}

# ── Tasks ────────────────────────────────────────────────────────────
# Topics for the style tasks are deliberately period-neutral (love,
# storms, ambition) so the modernism check measures craft, not topic.

USECASES: dict[str, list[dict]] = {
    "shakespeare-voice": [
        {"id": "sh-praise", "q": "Write 8 lines of verse in the style of Shakespeare, praising a beloved friend's loyalty.", "checks": ["archaic_density", "no_modernisms", "verse_shape"]},
        {"id": "sh-storm", "q": "Write 8 lines of verse in the style of Shakespeare describing a storm at sea.", "checks": ["archaic_density", "no_modernisms", "verse_shape"]},
        {"id": "sh-ambition", "q": "Write a short Shakespearean soliloquy (8-10 lines of verse) spoken by a general tempted to seize the crown.", "checks": ["archaic_density", "no_modernisms", "verse_shape"]},
        {"id": "sh-mourning", "q": "Write 8 lines of verse in the style of Shakespeare mourning a fallen soldier.", "checks": ["archaic_density", "no_modernisms", "verse_shape"]},
    ],
    "einstein-method": [
        {"id": "ei-simultaneity", "q": "Two friends standing far apart both say they clapped 'at the same moment'. Reason carefully, in the style of a physicist, about whether that claim has a definite meaning.", "checks": ["thought_experiment", "invariance", "operational"]},
        {"id": "ei-elevator", "q": "You wake in a sealed windowless room and feel weight pressing you to the floor. Reason about what you can and cannot conclude about your situation.", "checks": ["thought_experiment", "invariance", "limit_case"]},
        {"id": "ei-mountain-clock", "q": "Design a way to decide whether time passes at the same rate on a mountaintop and in a valley. Be rigorous about what 'time passes' means.", "checks": ["thought_experiment", "operational", "limit_case"]},
        {"id": "ei-hot-cold", "q": "Someone claims their new material 'feels colder than the room it is in'. Analyze the claim like a careful physicist.", "checks": ["thought_experiment", "operational", "invariance"]},
    ],
    "react-craft": [
        {"id": "re-userlist", "q": "Write a React component UserList that receives an array of users ({id, name, email}) as a prop and renders them as a list.", "checks": ["function_component", "map_has_key", "key_not_index"]},
        {"id": "re-gallery", "q": "Write a React component Gallery that renders a grid of photos from a prop photos: {id, src, caption}[].", "checks": ["function_component", "map_has_key", "img_alt"]},
        {"id": "re-signup", "q": "Write a React signup form component with name and email fields and a submit button.", "checks": ["function_component", "labelled_inputs", "no_div_onclick"]},
        {"id": "re-counter", "q": "Write a React counter component with increment and decrement buttons.", "checks": ["function_component", "functional_update", "no_div_onclick"]},
    ],
    "wordpress-expert": [
        {"id": "wp-form", "q": "Write a WordPress plugin snippet: an admin form that saves a custom 'support email' setting, and the handler that processes the form submission.", "checks": ["wp_nonce", "wp_capability", "wp_sanitizes_input", "wp_escapes_output"]},
        {"id": "wp-query", "q": "Write a WordPress function that looks up rows from a custom table wp_myplugin_orders by customer id and displays the order totals in HTML.", "checks": ["wp_prepared_query", "wp_escapes_output", "wp_prefix"]},
        {"id": "wp-shortcode", "q": "Write a WordPress shortcode [recent_books count=\"5\"] that lists recent posts of a custom post type 'book'.", "checks": ["wp_hooks", "wp_escapes_output", "wp_prefix"]},
        {"id": "wp-ajax", "q": "Write the WordPress code for an AJAX endpoint that lets a logged-in user dismiss a notice, storing the dismissal in user meta.", "checks": ["wp_hooks", "wp_nonce", "wp_sanitizes_input"]},
    ],
}


def ask(model: str, system: str, user: str) -> str:
    saved_sys = evaluate.SYSTEM
    evaluate.SYSTEM = system
    try:
        return evaluate.ask(model, user, None, num_predict=800)
    finally:
        evaluate.SYSTEM = saved_sys


def run_case(model: str, case: str, top_k: int) -> dict:
    tasks = USECASES[case]
    candidates = sorted(DIST.glob(f"{case}-*.ydbpack"))
    if not candidates:
        raise SystemExit(f"no built pack for {case} — run: python packs/build.py --all")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "host.db"), 64)
        pack_id = db.mount_pack(str(candidates[-1]))
        ctx = db.pack_context() or ""
        mounted_system = f"{BASE_SYSTEM}\n\n{ctx}" if ctx else BASE_SYSTEM

        rows = []
        for t in tasks:
            hits = db.recall_text(t["q"], top_k=top_k)
            retrieved = [
                h["text"]
                for h in hits
                if h.get("scores", {}).get("similarity", 0.0) >= MIN_SIMILARITY
            ]
            user_mounted = t["q"]
            if retrieved:
                joined = "\n".join(f"- {c}" for c in retrieved)
                user_mounted = (
                    f"Reference material from an attached knowledge pack:\n{joined}\n\n"
                    f"Task:\n{t['q']}"
                )

            base = ask(model, BASE_SYSTEM, t["q"])
            mnt = ask(model, mounted_system, user_mounted)
            per_check = {
                c: {"baseline": CHECKS[c](base), "mounted": CHECKS[c](mnt)}
                for c in t["checks"]
            }
            rows.append(
                {
                    "id": t["id"],
                    "checks": per_check,
                    "baseline_pass": sum(v["baseline"] for v in per_check.values()),
                    "mounted_pass": sum(v["mounted"] for v in per_check.values()),
                    "n_checks": len(per_check),
                    "retrieved": len(retrieved),
                    "baseline_answer": base.strip()[:600],
                    "mounted_answer": mnt.strip()[:600],
                }
            )
            marks = " ".join(
                f"{c}:{'-+'[v['baseline']]}{'-+'[v['mounted']]}" for c, v in per_check.items()
            )
            print(f"  {t['id']:<16} (retrieved {len(retrieved)})  {marks}")

        db.unmount_pack(pack_id)
        db.close()

    errors = sum(
        1
        for r in rows
        for k in ("baseline_answer", "mounted_answer")
        if r[k].startswith("<<error")
    )
    if errors:
        raise SystemExit(f"{errors} model calls failed — refusing to report a score.")

    total = sum(r["n_checks"] for r in rows)
    return {
        "model": model,
        "case": case,
        "tasks": len(rows),
        "checks_total": total,
        "baseline": sum(r["baseline_pass"] for r in rows),
        "mounted": sum(r["mounted_pass"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--case", action="append", default=[], choices=list(USECASES))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "efficacy-usecases.json")
    args = ap.parse_args()

    evaluate.OLLAMA = resolve_host(args.host)
    print(f"ollama: {evaluate.OLLAMA}")

    results = []
    for model in args.model or ["qwen3.5:4b"]:
        for case in args.case or list(USECASES):
            print(f"\n=== {model}  x  {case} ===")
            t0 = time.time()
            res = run_case(model, case, args.top_k)
            res["seconds"] = round(time.time() - t0, 1)
            results.append(res)

    print("\n" + "=" * 72)
    print(f"{'model':<15}{'use case':<22}{'checks':>8}{'baseline':>10}{'mounted':>9}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['model']:<15}{r['case']:<22}{r['checks_total']:>8}"
            f"{r['baseline']:>10}{r['mounted']:>9}"
        )
    print("=" * 72)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
