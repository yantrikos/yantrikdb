# Selective Facet Composition V1: Pre-Call Abort

The frozen v1 scope-similarity selector was run only through its product
preflight on the 400-row BEAM 100k cohort. No answer or judge calls were made.

## Result

- Canonical instruction targets retained: **36/40**
- Ordinary control contexts changed: **0/400**
- Extracted target facets missing from storage: **0/40**
- Decision: **abort v1 before external scoring**

The four misses were:

| Query ID | Query | Omitted rule |
|---|---|---|
| `10_instruction_following_0` | When was the Montserrat Writers' Festival? | Format dates as Month Day, Year for timeline details |
| `17_instruction_following_0` | When is my meetings at Montserrat Studios? | Format dates as MM/DD/YYYY for scheduling details |
| `17_instruction_following_1` | When was my meetings at East Janethaven Library? | Format dates as MM/DD/YYYY for scheduling details |
| `20_instruction_following_1` | When is the non-provisional patent filing scheduled? | Confirm exact dates for deadlines or meetings |

All omitted rules governed answer form. Query-to-scope similarity instead
favored facets whose topics appeared literally in the questions. Fixed-budget
checks of full-directive, action-only, combined action/scope, and dual-lane
ranking retained at most 36/40. Per the preregistration, `k` was not increased.

V2 therefore tests complete-lane additive composition while all verified
facets fit the preregistered 256-token budget. Budget-pressure selection for
larger stores remains a separate product problem.
