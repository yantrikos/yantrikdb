"""Compare gold-answer evidence coverage in two AMB context artifacts."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


_STOP = set(
    "the a an and or but if then of to in on at for with by from as is are was "
    "were be been being it its this that these those i you he she they we my your "
    "our their what which who how when where why do does did have has had not no "
    "yes so than too very can will just should now also".split()
)
_VALUE_RE = re.compile(
    r"(?:\$?\b\d[\d,]*(?:\.\d+)?%?|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)",
    re.IGNORECASE,
)


def _load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload if isinstance(payload, list) else payload["results"]
    return {row.get("query_id") or row.get("qid"): row for row in rows}


def _words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z][a-z0-9']{2,}", text.lower())
        if word not in _STOP
    }


def _coverage(gold: str, context: str) -> float:
    words = _words(gold)
    context = context.lower()
    return sum(word in context for word in words) / len(words) if words else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    shared = sorted(set(baseline) & set(candidate))

    word_before = []
    word_after = []
    values_before = values_after = values_total = 0
    categories = defaultdict(lambda: [[], [], 0, 0, 0])
    for query_id in shared:
        candidate_row = candidate[query_id]
        gold_answers = candidate_row.get("gold_answers")
        if not gold_answers:
            gold_answers = baseline[query_id].get("gold_answers") or []
        gold = " ".join(map(str, gold_answers))
        before = baseline[query_id].get("context") or ""
        after = candidate_row.get("context") or ""
        before_coverage = _coverage(gold, before)
        after_coverage = _coverage(gold, after)
        values = {value.casefold() for value in _VALUE_RE.findall(gold)}
        before_hits = sum(value in before.casefold() for value in values)
        after_hits = sum(value in after.casefold() for value in values)
        word_before.append(before_coverage)
        word_after.append(after_coverage)
        values_before += before_hits
        values_after += after_hits
        values_total += len(values)
        category = str(
            (baseline[query_id].get("meta") or {}).get("question_category")
            or query_id.split("_", 1)[1].rsplit("_", 1)[0]
        )
        aggregate = categories[category]
        aggregate[0].append(before_coverage)
        aggregate[1].append(after_coverage)
        aggregate[2] += before_hits
        aggregate[3] += after_hits
        aggregate[4] += len(values)
        if not args.summary_only:
            print(
                f"{query_id:34} words {before_coverage:.2f}->{after_coverage:.2f} "
                f"values {before_hits}/{len(values)}->{after_hits}/{len(values)}"
            )

    for category, aggregate in sorted(categories.items()):
        before_words, after_words, before_values, after_values, total = aggregate
        print(
            f"{category:28} words {statistics.fmean(before_words):.3f}"
            f"->{statistics.fmean(after_words):.3f} values "
            f"{before_values}/{total}->{after_values}/{total}"
        )

    print(
        f"mean word coverage {statistics.fmean(word_before):.3f}"
        f"->{statistics.fmean(word_after):.3f}"
    )
    print(
        f"value coverage {values_before}/{values_total}"
        f"->{values_after}/{values_total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
