"""wordpress-theme's craft definition — wraps the shared theme.json checker.

The interface every craft pack exposes: ARTIFACT, CHECKLIST, grade_text,
train_briefs, holdout_briefs, and optionally canonical() to normalise an
admitted artifact before it becomes training data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json

import wp_theme_checks as W

ARTIFACT = "theme.json (version 3)"

CHECKLIST = """MECHANICAL CHECKLIST — the validator tests these paths exactly:
- version: 2 or 3
- settings.typography.fluid: true
- settings.typography.fontSizes: 5-7 entries
- settings.layout.contentSize: 34-42rem (a measure, never 1200px); wideSize set
- settings.color.palette: >=4 entries, slugs named by role (base, contrast, primary, secondary)
- settings.spacing.spacingScale or spacingSizes (>=4 steps)
- settings.useRootPaddingAwareAlignments: true whenever styles.spacing.padding is set
- styles.typography.fontFamily, fontSize AND lineHeight (1.4-1.8) — declared presets must be APPLIED
- styles.elements.h1 (or h2) typography.lineHeight: 1.0-1.3
- styles.color.background and .text — never #000, #fff, black or white anywhere in base/contrast/background/text
- styles.elements.link.color.text set
- styles.spacing.blockGap set"""

grade_text = W.grade_text
train_briefs = W.train_briefs
holdout_briefs = W.holdout_briefs


def canonical(raw: str) -> str | None:
    theme = W.extract_json(raw)
    return json.dumps(theme, indent=2) if theme else None
