# Independent source-derived teaching gaps for Letterpress

This checklist was derived from `compiler.py`, `drive.py`, `photos.py`, `shoot.py`, and the hand-authored `.ops` examples only. The pack corpus, constitution, manifest, and author-written evaluation were not opened.

## Ranked checklist

1. **Critical — Ground every actionable claim in the brief.** Prices and clock times are acted upon by readers. Numeric forms absent from the brief cause the entire op line to be deleted, not merely softened. The model must copy supplied values exactly or write around missing values. This is especially important because written-out prices, dates, and many other factual claims are not mechanically caught.

2. **Critical — Never fabricate testimonials or endorsements.** The generation planner withholds the `quote` kind, but that is only a partial boundary: a made-up customer quotation placed in a legal `TEXT` or `ITEM` is not scanned and can survive. The model itself must understand that no named customer, attribution, review, or endorsement is available unless a human supplies it.

3. **Critical — Never invent an image URL.** The model's responsibility is a subject query, not asset identity. A `url=` argument is silently ignored by the current parser and can produce a default elevation drawing while strict compilation still reports clean. A URL disguised as `photo=` becomes search text and normally degrades to a strict-failing fallback. The safe learned behavior is to emit only a short `photo=` query or a valid `motif=` choice.

4. **Critical — Treat proof bands as evidence displays, not copy opportunities.** Every proof `ITEM` title must contain digits grounded in the brief, and at least two grounded items must survive or the whole section disappears. The model must not estimate, round, convert prose numbers into digits, import quantities that belong to another section, or relabel an arbitrary phrase as a figure. The current substring comparison can wrongly let `19` through when only `2019` is grounded, so exact copying remains necessary even where enforcement is imperfect.

5. **Critical — Obey the bounded-call protocol exactly.** Output must be one recognized op per line with key/value arguments and no prose, Markdown, code fences, or HTML/CSS/JavaScript. Non-op text is removed but counts as noise and fails conformance. During a scoped section call, repeated `SITE`, `THEME`, or `SECTION` lines and content aimed at another `sec` are dropped as stray and also fail conformance.

6. **High — Choose a genre before choosing the middle of the page.** The generic section menu otherwise collapses very different businesses into the same features/detail/FAQ skeleton. The model must know the genre-specific topology and the information that makes that genre's page useful, while omitting a section whose required facts are absent.

7. **High — Restaurant pages are visit tools.** Put opening hours and street address high in a `note`; use `features` for only a handful of representative dishes. Do not manufacture missing hours, prices, or a complete menu.

8. **High — Recipe section names have private meanings.** `roster` means ingredients with the brief's exact quantities, `features` means ordered method steps with one action per item, and `proof` means yield, working/proving/baking times, and temperature. Generic services copy in any of these slots breaks the page's purpose.

9. **High — Portfolio pages are short evidence, not a sales funnel template.** Use `features` for selected work with material, size, and client context supplied by the brief. Do not add FAQ, stats/proof, generic services, process filler, or selling language.

10. **High — Other genres also remap common kinds.** Travel features are an itinerary and proof is booking qualifiers; event proof is date/place/venue/ticket facts; charity proof explains where money goes; shop and property proof carry specifications; a course's features describe what students can do; a trade's detail names the covered area. These mappings are not inferable from the op names alone.

11. **High — Decide photograph versus motif from observability.** Food, rooms, landscapes, products, properties, and places call for photographs. Software, services, systems, time spans, coverage, and other non-photographable ideas call for compiler-drawn motifs. Confusing the two yields an honest but visibly wrong page.

12. **High — Photo queries are terse retrieval keys, not art direction.** Use two to four plain subject nouns. Mood language and descriptive sentences overconstrain an AND-style search. The resolver strips stopwords, progressively relaxes shape/size constraints and query terms, rejects unusable licences, undersized files, unknown formats, and lexically irrelevant hits, then prefers contemporary stock-like results. Resolution failure becomes a motif fallback and makes a strict build fail.

13. **High — Motifs have semantic jobs and should not repeat.** Elevation is premises/property; topography is land or spread; schematic is systems; dial is measurement; specimen is typography/language; strata is duration/history; lattice is craft/material; orbit is reach/community. Reusing the hero motif in detail triggers harness substitution from a family shortlist, which can change the model's intended art.

14. **High — Know every section's legal slots and `ITEM` schema.** `TEXT` and `ACTION` with illegal slots are rejected. In proof, `ITEM.title` is the figure; in FAQ it is the question; in roster it is a service/role or genre-specific entity; in ordinary features it is the offered thing. An op may parse while saying the wrong kind of thing, so semantic slot knowledge is load-bearing.

15. **High — Write enough copy for the renderer's composition.** Feature bodies need two concrete sentences; FAQ answers need two or three; detail ledes need two or three; notes need exactly a title and lede; hero ledes should be full sentences. Thin content leaves layouts looking unfinished even when all mechanical gates pass.

16. **High — Close with the next decision, not a repeated headline.** The CTA title must advance from the hero toward when to come, what happens first, or what it costs if supplied. Repeating the hero is mechanically legal but makes the page read as filled-in.

17. **Medium — Inline emphasis uses one bracketed phrase from the title itself.** There is no `mark=` argument. The compiler accents only the first valid bracketed span and strips leftover brackets. The model should mark two or three existing title words once, not invent an external substring.

18. **Medium — Keep visual-system decisions at the correct boundary.** The model chooses family, mode, density, and one accent seed. The harness owns section layout and tone; the compiler owns type scale, spacing, full palette, contrast repair, responsive collapse, SVG, focus treatment, and motion fallback. Attempts to specify CSS, breakpoints, arbitrary colours, or pixel values are outside the language.

19. **Medium — Understand clean compilation versus usable output.** The compiler can still write a partial HTML artifact before strict mode returns nonzero for issues or recorded fallbacks. Conversely, unknown extra arguments are generally ignored and may remain falsely clean. A model should not interpret “an HTML file exists” as protocol conformance.

20. **Medium — Do not volunteer media descriptions as facts.** The harness strips model-authored `alt=` from generated MEDIA because invented captions previously described photographs that the compiler had not drawn. Motif captions are derived from the selected drawing; photo identity, licence, dimensions, credit, and source come from the resolver. A human-authored ops file has a different trust boundary.

21. **Medium — Mechanical render gates are not aesthetic or semantic judges.** The shooter checks settled desktop/mobile content, horizontal overflow, painted contrast including opacity, readable text size, one H1, and invisible text. Passing does not establish beauty, originality, brand fit, truth, or image relevance, and the model must not turn a browser receipt into those claims.

## Marketplace legitimacy

As currently described, a pack that teaches only this private op language while the required compiler is not distributed is inert packaging, not a complete marketplace capability. Calling out the dependency is honest but insufficient: a buyer still cannot execute the learned behavior. It becomes a legitimate add-on listing only if the marketplace can install or reliably obtain a compatible compiler, pins the protocol/compiler version, and evaluates the combined runtime; otherwise the listing should be bundled with the compiler or withheld.
