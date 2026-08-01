# letterpress corpus — driving the site compiler

Every record here is a worked example: the ops as you would actually
write them, and the reason the shape is what it is. The reasons are not
decoration. Most of them are a defect that shipped, was rendered, and
was looked at — the kit's rules are almost all scar tissue, and a rule
whose reason you know is one you can apply to a case it does not
literally cover.

## letterpress ops: the seven lines and what each one does

The whole language is seven operations. SITE names the business and
fills the masthead and footer. THEME chooses the typographic family, one
accent colour, light or dark, and the spacing density. SECTION opens a
band of the page and gives it an id, a kind, a layout and a tone. TEXT
puts words into a named slot of a section. ACTION makes a button. MEDIA
asks for a drawing. ITEM adds one entry to a section that carries a
list, and is the only op with two content arguments of its own.

Each line is a key-value list. Values with spaces take double quotes;
values without them do not need any. Order matters only in that a
section must be opened before anything can refer to it: SITE and THEME
appear once at the top, and every other op hangs off a SECTION through
its `sec=` id.

Nothing else is an op. A line of explanation, a markdown heading, a code
fence, or a comment is discarded and counted against the generation, so
a response that opens "Here are the ops:" has already lost a line.

```
SITE     name="Tide Mill"  nav="Bread,Visit,Contact"  location="Bristol"  contact="hello@tidemill.co"
THEME    family=editorial  accent=#b8442f  mode=light  density=roomy
SECTION  id=s1  kind=hero  layout=split  tone=quiet
TEXT     sec=s1  slot=eyebrow  text="Weekly loaf"
TEXT     sec=s1  slot=title    text="Bread [worth the walk]"
TEXT     sec=s1  slot=lede     text="Two ovens and no cafe. We bake twice a week and sell out by noon."
ACTION   sec=s1  slot=primary    label="Collection times"  href="#times"
ACTION   sec=s1  slot=secondary  label="This week's loaf"  href="#loaf"
MEDIA    sec=s1  slot=figure  motif=lattice
ITEM     sec=s2  slot=item  title="Tuesday"  body="Rye and caraway, forty loaves. Mixed by hand and proofed overnight in the cool cellar."
```

Note the shape of `TEXT`: the slot is the *value* of `slot=`, and the
words go in `text=`. Writing `eyebrow="..."` or `title="..."` as the key
is the commonest first mistake. `ITEM` is the exception — it really does
take `title=` and `body=`, because an item has two parts of its own.

## letterpress: never write HTML or CSS, and why the split is where it is

The compiler owns type scale, spacing rhythm, the twelve-column grid,
the entire palette, contrast, responsive collapse, and every SVG. You
own what the page says. This is not a stylistic preference; it is the
measured result.

The same model, same briefs, same number of calls, same design guidance
in prose, writing HTML directly: **failed every render gate on every
brief**. Writing the whole page in one call instead of five: failed
every one again. Both arms were handed "body text must reach at least
4.5:1 contrast" verbatim, and both produced text at 1.00:1 — the same
colour as its own background — because a model cannot evaluate a
contrast ratio. It can only assert one.

Split any brief's constraints into what needs *judgement* and what needs
*computation*. Judgement is yours. Computation belongs to a compiler.

## letterpress THEME: choosing a family and a single accent

```
THEME  family=editorial   accent=#b8442f  mode=light  density=roomy
THEME  family=studio      accent=#3b2fb8  mode=light  density=normal
THEME  family=technical   accent=#0f4c75  mode=dark   density=tight
```

- `editorial` — serif, magazine. Food, craft, architecture, writing,
  archives, clinics, anything with a physical premises or a history.
- `studio` — heavy sans, poster. Design, portfolios, campaigns, agencies.
- `technical` — monospace, schematic. Developer tools, hardware, data.

Pick the accent hue from the subject, then stop. The compiler solves
every other colour against its own ground; a mid-green that could not
carry a legible button label at any lightness gets walked darker until
it can. Naming a second colour anywhere is always wrong.

`density` scales the spacing rhythm only: `tight` for dense technical
pages, `roomy` for editorial pages that want air.

## letterpress hero: the opening section, with a drawing

The hero is the first band of the page and the only one that carries the
h1. It holds an eyebrow that places the business, a headline that makes
a claim, a lede of two sentences that supports it, two actions — the
thing you want the reader to do and the thing they will probably want
first — and one drawing.

The headline is where the emphasis brackets belong, on two or three of
its own words. Give the eyebrow the plain identifying fact, the sort of
line that would sit under the name on a sign, and let the headline be
the argument. A hero whose headline is just the business name and its
city has spent the largest type on the page saying nothing.

The lede is two full sentences. This is the most common place a
generated page gives itself away: a fragment under a large headline
leaves the top of the page looking like a placeholder.

```
SECTION  id=s1  kind=hero  layout=split  tone=quiet
TEXT     sec=s1  slot=eyebrow  text="Physiotherapy in Leeds"
TEXT     sec=s1  slot=title    text="Rehab that ends with you [back on the pitch]"
TEXT     sec=s1  slot=lede     text="Two practitioners and forty minutes an appointment. You see the same one from assessment through to discharge."
ACTION   sec=s1  slot=primary    label="Book an assessment"  href="#book"
ACTION   sec=s1  slot=secondary  label="What we treat"       href="#treat"
MEDIA    sec=s1  slot=figure  motif=elevation
```

`layout=split` puts the text on a 1–8 column span and the drawing on
7–13, so the headline crosses the gutter and overlaps the art's column.
A symmetric fifty-fifty split with everything vertically centred is what
made the first version read competent and inert.

The lede is two full sentences. A fragment here is the single biggest
contributor to a page looking unfinished.

## letterpress features: several things offered, as a numbered list

The workhorse section. It holds three to five things the business
offers, each a short title and a body of two sentences, introduced by
an eyebrow and a heading of its own.

What it actually contains depends on the genre, and that is worth being
deliberate about: on a trip page it is the itinerary, one item per day;
in a recipe it is the method, one action per step; for a publication it
is the recent pieces; for a course it is the curriculum; for a
portfolio it is the work itself. The section is the same shape and the
content is not interchangeable.

Two sentences per item is the rule that matters. The first says what
the thing is and the second gives one concrete detail — a duration, a
material, a limit, a step. One-line bodies are the commonest reason a
finished page still looks like a draft.

```
SECTION  id=s4  kind=features  layout=list  tone=quiet
TEXT     sec=s4  slot=eyebrow  text="How it runs"
TEXT     sec=s4  slot=title    text="Three appointments, then a plan you own"
ITEM     sec=s4  slot=item  title="Assessment"  body="Forty minutes. We test the joint, watch you move, and write down a baseline you can be measured against later."
ITEM     sec=s4  slot=item  title="Loading"     body="Four to six sessions, spaced to match tissue healing rather than a billing cycle."
ITEM     sec=s4  slot=item  title="Discharge"   body="You leave with the programme and the progressions. We tell you the criteria for returning to sport."
```

`layout=list` renders numerals on hairline rules; `layout=grid` renders
bordered cards. Prefer the list. Cards read as a component library and
are the most obviously generated element after a flat colour block.

Each body is **two sentences**: what it is, then one concrete detail — a
duration, a material, a step, a limit.

## letterpress faq: the only section that carries real paragraphs

```
SECTION  id=s5  kind=faq  layout=stack  tone=bold
TEXT     sec=s5  slot=eyebrow  text="Common questions"
TEXT     sec=s5  slot=title    text="What to expect from your first visit"
ITEM     sec=s5  slot=item  title="How long does a typical session last?"  body="Assessment and treatment are capped at forty minutes, though complex post-operative cases may extend slightly. We do not rush through protocols; every movement pattern is verified before progressing to load."
ITEM     sec=s5  slot=item  title="Do I need a gym membership?"           body="No. All the equipment is provided in the clinic space at no additional cost, and there is no contract at the end of your course of treatment."
```

Questions written the way a customer would type them, answers of two or
three full sentences with the specifics. This is the section people
actually read, and the one that lets a page carry prose rather than
captions — which is most of what separates a thin page from a real one.

It is also where invented facts appear most freely, because an answer
feels obliged to be concrete. Everything in an answer must come from the
brief.

## letterpress proof: a stats band, and the figures you may not invent

```
SECTION  id=s2  kind=proof  layout=stack  tone=bold
TEXT     sec=s2  slot=eyebrow  text="Since 1957"
TEXT     sec=s2  slot=title    text="What the collection holds"
ITEM     sec=s2  slot=item  title="12,400"     body="Reels of regional television."
ITEM     sec=s2  slot=item  title="1957"       body="The earliest surviving broadcast."
ITEM     sec=s2  slot=item  title="4K"         body="The resolution everything is digitised at."
ITEM     sec=s2  slot=item  title="3 days"     body="Open to researchers each week, free of charge."
```

`title=` is the figure, exactly as the brief writes it. `body=` says
what it counts. Two figures minimum — one stat is a row with its other
cells missing.

**If the brief states no figures, write no proof section.** Asked for a
stats band on a brief with no numbers in it, a 4B produced "140ms — time
to scan 5GB", "$3.8k/yr — cost savings" and "99% — accuracy". Three
fabricated benchmarks with the authority of a measurement, on a page a
real project would publish. An empty response is the correct answer.

## letterpress detail: the one thing a competitor could not copy

The detail section takes a single claim and spends real words on it. It
is the answer to "why this one and not the one down the road", and it
wants the thing that would be genuinely hard for a competitor to
replicate: a decade-old culture, a protocol written by the surgeon
rather than a template, reading a live catalogue instead of exporting a
dump. Generic superiority — better service, higher quality, more care —
is what every competitor also claims, and belongs nowhere.

Its lede runs two or three sentences, longer than the hero's, because
this is the place a reader who is already interested slows down.

The section carries the page's second drawing, and it must not be the
same motif as the hero's.

```
SECTION  id=s3  kind=detail  layout=split  tone=quiet
TEXT     sec=s3  slot=eyebrow  text="The secret"
TEXT     sec=s3  slot=title    text="Our starter is older than the [second oven]"
TEXT     sec=s3  slot=lede     text="It has been fed daily since before the shop had a counter. We do not buy yeast; we keep a living culture that remembers every season it has been through, and that is what the tang is."
ACTION   sec=s3  slot=primary  label="How we bake"  href="#bake"
MEDIA    sec=s3  slot=figure  motif=strata
```

Mirrors the hero: same twelve-column grid, columns reversed, art on the
left. That reversal is what gives a page rhythm instead of a stack of
identical rows — and it is why the motif here must differ from the
hero's.

Pick the genuinely distinguishing thing. "We care about quality" is not
it; a ten-year-old starter is.

## letterpress roster: the people or the services, without inventing names

A roster lists who the business is for, or what it offers, as a set of
short named entries. It suits a clinic listing the kinds of treatment,
a bakery listing the trades it supplies, or a tool listing the audiences
that use it. Three entries is the usual number; each gets a name and two
sentences saying what it covers and who it is for.

It reads as a directory rather than an argument, which is what makes it
useful mid-page: it answers "is this for me" without asking the reader
to follow a case. Put it after the section that makes the claim and
before the one that answers objections.

The trap is the heading. A roster invites a list of staff, and a list of
staff invites names, and a name is a fact about a real person you cannot
possibly have. Keep every entry a role or a service.

```
SECTION  id=s4  kind=roster  layout=stack  tone=quiet
TEXT     sec=s4  slot=eyebrow  text="Who we serve"
TEXT     sec=s4  slot=title    text="Bakers, butchers and home cooks"
ITEM     sec=s4  slot=item  title="Commercial bakeries"  body="We supply weekly batches to three shops in the West Country. Minimum order is fifty loaves."
ITEM     sec=s4  slot=item  title="Butcher shops"        body="Our rye pairs with cured meats for weekend markets. We deliver on Fridays before the counter opens."
```

`title=` is a **role or a service**, never a person's name. A named
individual is something you cannot know and should not assert — the same
refusal as the testimonial rule, applied one section over.

## letterpress note: one wide statement, used as punctuation

```
SECTION  id=s6  kind=note  layout=stack  tone=inverted
TEXT     sec=s6  slot=title  text="The last loaf leaves at [noon]"
TEXT     sec=s6  slot=lede   text="There is no afternoon bake and no holding back stock for later callers."
```

No eyebrow, no button, no drawing. It exists to break two dense sections
apart, and gains nothing from being given more. Use at most one per
page, and put the single most useful fact in the brief in it.

## letterpress cta: the closing section must say something new

```
SECTION  id=s7  kind=cta  layout=stack  tone=quiet
TEXT     sec=s7  slot=title  text="Assessments run Tuesday to Saturday"
TEXT     sec=s7  slot=lede   text="Great George Street, five minutes from the station."
ACTION   sec=s7  slot=primary  label="Book an assessment"  href="#book"
```

The commonest way a generated page reads as filled-in rather than
written is a closing headline that restates the hero headline. Say the
next thing instead: when to come, what happens first, what it costs the
reader in time.

`layout=stack` renders the heading left and the action pushed right.
Centring everything is the safe choice and reads as a template.

## letterpress slots: which slots each section kind will accept

A slot is the role a line plays inside its section, and the set of legal
roles depends on the kind. The compiler checks the pair and rejects an
illegal one rather than quietly dropping it into the nearest sensible
place, because a coerced slot produces a page that looks fine and says
something you did not write.

A hero accepts an eyebrow, a title, a lede, a primary and a secondary
action, and a figure. Features accepts an eyebrow, title, lede and item.
Proof accepts an eyebrow, title and item, but no lede and no action — a
band of figures does not argue, it states. Detail accepts an eyebrow,
title, lede, a single primary action and a figure. Faq and roster accept
an eyebrow, title, lede and item. Note accepts only a title and a lede,
which is the whole point of it. Cta accepts an eyebrow, title, lede and
both actions, but never a figure. Quote accepts only a quote and an
attribution.

The two mistakes this catches are asking for an item in a section that
has no list — a hero or a cta — and asking for an action in a proof or a
faq, where the reader is reading rather than deciding.

## letterpress: two figures on one page must be different drawings

If a page has both a hero figure and a detail figure, the two motifs
must differ. Choose the second from what that section is actually about,
not from the same instinct that produced the first.

This is the most visible sameness defect the kit has had. Across twelve
generated pages, ten used the identical motif for both figures — the
drawing at the top of the page and the drawing halfway down were the
same picture, which makes a long page feel like a short page repeated.
Half of all figures were the building, because it is the first row of
the table and a small model reads the first row as the default.

The fix in your hands is to pick from the subject. An archive's hero is
its span of years and its detail is the thing it holds; a bakery's hero
is the craft and its detail is the age of the starter. Those are two
different drawings before you have thought about variety at all.

## letterpress: never attribute words to a person or a company

Do not write a testimonial, a review, a quoted endorsement, or a named
individual described as staff or as a customer. Not in a quote section,
not in a roster item, not inside a faq answer.

The reason is worth keeping straight, because it is not squeamishness. A
figure can be checked: either the number is in the brief or it is not,
and the kit strips it if it is not. A testimonial cannot be checked even
in principle. The brief contains no customers, so every possible
sentence you could write is invented — there is no version of the task
where you get it right.

Asked for one, a 4B produced "Sarah Chen, DevOps Lead at RiverScale": a
named person at a named company, neither of which exists, attesting to a
product they have never used, on a page that would go up under someone's
real business name. When a slot cannot be verified even in principle,
the honest response is to leave it to a human who is answerable for the
words. That is what `quote` is for, and why nothing generates one.

## letterpress: write copy that is specific, with real names and real detail

Write specific copy. Real names, real detail, the thing you would tell
someone standing in the doorway. Never "Lorem ipsum", never "Your
Company", never "Feature One", and never a headline that would fit any
business in the same trade.

Specific means a detail that could only be true of this business: a
duration, a material, a street, a step in a process, a limit someone
actually lives with. "We pride ourselves on quality" is plausible about
anyone and therefore says nothing. The test is whether the sentence
would survive being read by the owner: "rye and caraway, forty loaves,
mixed by hand and proofed overnight in the cool cellar" is specific,
"artisanal breads crafted with passion" is filler and reads as filler
even to someone who cannot say why.

## letterpress: specific is not the same as invented

Reaching for a concrete-sounding number is how a page acquires facts
nobody can stand behind, and it happens precisely because the writing
advice above is working — a sentence feels weak, so it gets a figure
attached to make it land.

The line is this: elaborate what the brief implies, never what it does
not say. A bakery with two ovens and no cafe implies a counter, a queue,
and bread running out. It does not imply a price, an opening hour, a
number of loaves, or a percentage. A physiotherapy clinic with two
practitioners implies continuity of care. It does not imply a session
length or a fee.

Anything a reader would act on — money, times, measured performance —
must come from the brief verbatim or not appear at all.


## letterpress motifs: which drawing for which subject

The compiler draws eight abstract line diagrams. They are never
photographs and never the object itself, so choose by what the diagram
*is*, and let the compiler caption it.

    elevation   a building face-on with a dimension line.
                ONLY premises and property — a place someone visits.
    topography  nested contour rings. Land, environment, coverage.
    schematic   nodes on a grid joined by routed runs. Software,
                infrastructure, pipelines, systems.
    dial        a graduated gauge with a needle. Instruments,
                precision, measurement.
    specimen    a large letterform over baseline rules. Typography,
                publishing, language.
    strata      stacked bands cut by a section line. A span of time —
                a record, a back catalogue, a history, anything
                defined by the years it covers.
    lattice     an over-under weave. Craft and material — food,
                textile, joinery, print, made by hand in batches.
    orbit       concentric rings with nodes. Reach and membership —
                services, a community, coverage around a centre.

`elevation` is not the default. Half of all figures across twelve
generated pages came out as the building — a house for a bakery, a
clinic and a film archive alike — because it is the first row. A bakery
is `lattice`. An archive running 1957 to 1994 is `strata`. A Postgres
CLI is `schematic`.

## letterpress: do not write the figure's caption

`MEDIA` takes `slot=figure` and `motif=`. It does not take `alt=`.

The compiler drew the picture and is the only party that knows what is
in it, so it writes the caption. Given the chance, a model captions the
drawing as though it were a photograph: "A reel of film on a turntable"
over a line drawing of a building, "Terminal output showing index names"
over a gauge. That survived per-motif descriptions and a worked example
for each, which is the signal that the decision belonged elsewhere.

## letterpress: planning which sections a page needs

A page opens `hero` and closes `cta`. Choose three or four between them
from `proof`, `detail`, `features`, `faq`, `roster`, `note`, and choose
for the business in front of you.

```
a bakery            hero  features  roster  detail  faq   cta
a physio clinic     hero  features  detail  roster  faq   cta
an open-source CLI  hero  features  detail  faq     cta
a film archive      hero  proof     detail  features note cta
```

Only the archive brief states figures, so only it gets `proof`. The CLI
has no premises and no staff to list. Include `detail` (or another
section carrying a drawing) always, or the page becomes an unbroken
column of prose — which is what happens when the copy gets richer and
nothing was planned to break it up.

## letterpress tone: alternate the grounds

`tone=quiet` is the page ground, `bold` a tinted panel, `inverted` a
dark band. Alternate them so consecutive sections do not merge: a
six-section page with every tone `quiet` reads as one continuous column
however different the sections are. Use `inverted` sparingly — once per
page, for a `note` or a closing band.

## letterpress: what the compiler rejects, and the errors that produce it

- **A slot that does not belong to the kind.** `slot=item` in a `hero`,
  `slot=primary` in a `features`. Rejected, never coerced.
- **An unknown `kind`, `layout`, `tone` or `motif`.** Closed
  vocabularies; a near-miss like `motif=building` is not resolved to
  `elevation`.
- **`sec=` naming a section that does not exist yet.** Ops must follow
  their `SECTION`. Writing ahead into a section you have not been asked
  for produces orphans.
- **A repeated `SITE` or `THEME`.** The first one wins; a second is the
  model overstepping, and a later `THEME` silently overwriting the first
  is exactly the kind of quiet wrongness the kit refuses.
- **Any non-op line.** Prose, markdown, a code fence, an explanation.

The compiler reports every substitution it makes rather than quietly
rescuing a malformed script, because a generous compiler produces a
handsome page from a failed generation and you would be reading the
compiler's work as the model's.

## letterpress genres: every kind of site wants a different page

The section menu is generic and a business is not. Left to choose
freely from it, a model plans features, then detail, then faq, for a
bakery and a database tool alike — the menu offers no reason to prefer
one shape over another, so it converges on the middle.

Genre supplies that reason. A restaurant page lives or dies on hours
and address; a portfolio must not carry an FAQ; a tour is worthless
without its duration and what is included; a recipe is a list of
ingredients and a numbered method and very little else. These are not
style preferences. They are what the reader came for, and a page that
omits them has failed at the only job it had.

Each record below gives a plan that works, the content that genre
cannot omit, and the mistake it invites. Follow the one that matches
the brief; they are starting points, not templates, and a brief that
clearly wants a different shape should get one.

## letterpress genre: restaurant, cafe or bar

    hero  ->  note  ->  features  ->  detail  ->  faq  ->  cta
    editorial family, photo-led

Someone reading a restaurant site wants three facts quickly: where it
is, when it opens, and whether they can get a table tonight. Put the
hours and the street in a `note` directly under the hero rather than in
the footer, where a phone reader will never scroll to find them.

`features` is a handful of dishes that show the kitchen's range, never
the full menu — that belongs on its own page. `detail` is the sourcing,
the room, or the person who cooks. `faq` answers booking, walk-ins,
dietary requirements and children.

MUST CARRY: street address, opening days and hours, how to book.
NEVER WRITE THESE — the genre tempts you to and you must not: invented prices and opening times. Both must come from the
brief; a reader turns up with the wrong money at the wrong hour.
PHOTOGRAPHS: the room and the food.

## letterpress genre: travel, tours and guided trips

    hero  ->  proof  ->  features  ->  detail  ->  faq  ->  cta
    editorial family, photo-led

Nobody books a trip without knowing how long it is, how many people
come, and what the price includes. `proof` carries exactly that — days,
group size, departure months, distance covered — and it sits high on
the page because it is the qualifying information.

`features` is the itinerary, one item per day or per stage, each with a
concrete detail: a distance, a crossing, a place slept. `detail` is
what separates this operator from the twenty others running the same
route. `faq` covers fitness, visas, weather, insurance, and what
happens when conditions turn.

MUST CARRY: duration, group size, what is and is not included.
NEVER WRITE THESE — the genre tempts you to and you must not: a price, a departure date and an altitude that were never in
the brief.
MOTIF when there is no photograph: `topography`.

## letterpress genre: personal portfolio

    hero  ->  features  ->  detail  ->  cta
    studio family, short

A portfolio is the shortest page in this list. The work is the
argument, so the page gets out of its way: a hero saying plainly who
you are and what you make, the work itself, one honest paragraph on how
you approach it, and a way to reach you.

The hero headline is your name and your discipline, not a slogan.
"Sculptural furniture in oak and steel" beats "Crafting tomorrow's
experiences" by the distance between a fact and a noise.

MUST CARRY: what you do, examples of it, how to make contact.
NEVER: an FAQ, a stats band, or a roster of "services" — a portfolio
with a testimonial section reads as a small agency pretending to be a
larger one.
LENGTH: three or four sections. Padding it is the commonest mistake.

## letterpress genre: blog or publication

    hero  ->  features  ->  detail  ->  cta
    editorial family

The hero states the subject and the point of view: what this publishes
and why it deserves a reader's attention. `features` is the recent
pieces, each item a title and one sentence making the case for reading
it — never a truncated first paragraph. `detail` is who writes it and
what they know. `cta` is the subscription, and it should say what
arrives and how often.

MUST CARRY: what the publication covers, recent pieces, how to
subscribe.
NEVER WRITE THESE — the genre tempts you to and you must not: a subscriber count and a publishing frequency the brief never
stated.

## letterpress genre: recipe

    hero  ->  proof  ->  roster  ->  features  ->  faq  ->  cta
    editorial family, photo-led

A recipe has the strictest structure of any genre here, because a cook
reads it standing up with wet hands.

`proof` is the numbers, at the top where they can be found: serves,
prep time, cook time, oven temperature. `roster` is the INGREDIENTS,
one item each, quantity in the title and preparation in the body —
"300g strong white flour" / "sifted, plus extra for dusting".
`features` is the METHOD, numbered, one step per item, each step a
single action. `faq` is substitutions, making ahead, and storage.

MUST CARRY: quantities, times, temperature, yield.
NEVER: a step that does two things. Split it.
NEVER WRITE THESE — the genre tempts you to and you must not: quantities and temperatures that were not in the brief — the
one genre where an invented number does not merely mislead a reader, it
ruins the dish.

## letterpress genre: local trade — plumber, electrician, builder

    hero  ->  features  ->  detail  ->  proof  ->  faq  ->  cta
    editorial or technical family

Someone on this page has a problem now. The hero says what you do and
where you cover, in that order. `features` is the services, plainly
named — "boiler repairs", not "heating solutions". `detail` is the area
covered, with the towns named. `proof` is years trading, callout time,
registration numbers. `faq` answers how quotes work, whether callouts
are charged, and what counts as an emergency.

MUST CARRY: the services, the area covered, how to make contact.
NEVER WRITE THESE — the genre tempts you to and you must not: a response time and a callout fee that were never stated.
MOTIF: `topography` for the area covered, `orbit` for reach.

## letterpress genre: clinic or practitioner

    hero  ->  features  ->  detail  ->  roster  ->  faq  ->  cta
    editorial family

A patient wants to know what you treat, what an appointment is like,
and what it costs them in time. `features` is what you treat. `detail`
is your method — the thing a competitor could not copy. `roster` is the
services or disciplines, never named individuals. `faq` is the section
people actually read: the first visit, session length, referrals,
confidentiality.

MUST CARRY: what is treated, what a first appointment involves, how to
book.
NEVER WRITE THESE — the genre tempts you to and you must not: session fees and appointment lengths not in the brief.

## letterpress genre: developer tool or open-source project

    hero  ->  features  ->  detail  ->  faq  ->  cta
    technical family, motif=schematic

The hero says what it does in one line a developer can evaluate — the
verb and the object, not the category. "Finds unused indexes in
Postgres" beats "database optimisation platform".

`features` is capabilities, each with a concrete mechanism. `detail` is
the architectural decision that makes it different: what it reads, what
it never touches. `faq` is permissions, data handling, CI, and what
happens when something is unavailable. `cta` is install or docs.

MUST CARRY: what it does, how it is installed, the licence.
NEVER WRITE THESE — the genre tempts you to and you must not: benchmarks. "Scans in under ten seconds", "handles 50TB",
"ninety percent of cases" — every one of those is a measurement that
nobody took.

## letterpress genre: charity or nonprofit

    hero  ->  proof  ->  detail  ->  features  ->  note  ->  cta
    editorial or studio family

The hero states the problem, not the organisation. `proof` is where the
money goes, which is the question every donor has and few charity pages
answer plainly. `detail` is the work itself, concretely. `features` is
the ways to help. `note` is the reassurance: no phone calls, cancel
whenever, the list is never sold.

MUST CARRY: what the money does, how to give, how to stop giving.
NEVER WRITE THESE — the genre tempts you to and you must not: percentages, beneficiary counts and overhead ratios that were
never in the brief. This is the genre where an invented figure does the
most reputational damage.

## letterpress genre: event or conference

    hero  ->  proof  ->  features  ->  roster  ->  faq  ->  cta
    studio family

`proof` is the date, the city, the venue and the ticket price, because
those four decide whether anyone reads further. `features` is the
programme. `roster` is the sessions or strands — speakers are named
people, so name them only if the brief does. `faq` is access, travel,
refunds and what a ticket includes.

MUST CARRY: date, place, how to get a ticket.
NEVER WRITE THESE — the genre tempts you to and you must not: dates. Never invent one. A wrong date is the single most
damaging error this genre can carry.

## letterpress genre: shop or single product

    hero  ->  detail  ->  proof  ->  features  ->  faq  ->  cta
    studio family, photo-led

`hero` is the product photographed and one line saying what it is.
`detail` is why it was made this way. `proof` is the specification:
dimensions, weight, materials, warranty. `features` is what it does or
what is in the box. `faq` is shipping, returns and sizing, which is
where most abandoned purchases are lost.

MUST CARRY: what it is, what it costs, how it ships, how to return it.
NEVER WRITE THESE — the genre tempts you to and you must not: a price, a delivery window and dimensions from nowhere.

## letterpress genre: studio, agency or practice

    hero  ->  features  ->  detail  ->  cta
    studio family

`hero` is the positioning: the kind of work you take and who for.
`features` is the services. `features` is selected work, which for a
design studio is the whole argument. `detail` is how you work — the
process, the size of an engagement, what a first month looks like.

MUST CARRY: what you do, examples, how to start a conversation.
NEVER: a testimonial you invented, or client names you cannot verify.

## letterpress genre: property or a place to let

    hero  ->  proof  ->  features  ->  detail  ->  cta
    editorial family, photo-led, motif=elevation

`proof` is bedrooms, floor area, price and council tax band. `features`
is the rooms, one item each with its dimensions. `detail` is the area:
the schools, the station, how long it takes to walk there. This is the
one genre where `elevation` is exactly the right drawing, because the
subject is literally a building.

MUST CARRY: price, size, location, how to arrange a viewing.
NEVER WRITE THESE — the genre tempts you to and you must not: every number on the page. Dimensions, prices and distances
must come from the brief verbatim or not appear.

## letterpress genre: school, course or training

    hero  ->  features  ->  proof  ->  roster  ->  faq  ->  cta
    editorial family

`features` is the curriculum, one item per module or term, each saying
what a student can do afterwards rather than what is covered. `proof`
is duration, contact hours, class size and cost. `roster` is the
subjects or the departments. `faq` is entry requirements, fees,
funding, and what happens if someone falls behind.

MUST CARRY: what is taught, how long it takes, what it costs, how to
apply.
NEVER WRITE THESE — the genre tempts you to and you must not: pass rates, employment statistics and class sizes that were
never stated.

## letterpress stock images: how to put a photograph on the page

A photograph on a letterpress page comes from a stock image search that
the compiler runs for you. You describe the picture; you never supply
an image, an image URL, or a file name.

MEDIA takes either `photo="..."` or `motif=...`, never both. A photo
query is resolved against a licensed image library, downloaded, and
embedded by the compiler with its real dimensions and its credit line.

```
MEDIA  sec=s1  slot=figure  photo="sourdough bread bakery"
MEDIA  sec=s3  slot=figure  motif=strata
```

You never write a URL. You cannot know that any particular image
exists, is licensed for use, or shows what its filename claims — an
invented address is a broken picture at best and somebody else's
premises presented as yours at worst. Naming the subject is judgement,
which is yours. Finding a real file for it is lookup, which is not.

Use a photograph when the subject is something a camera can see: food,
a room, a landscape, a made object, a place. Use a drawing when it is
not: software, a service, a span of years, an area covered. A
restaurant, a trip, a recipe, a property and a product want
photographs. A developer tool and a coverage map do not.

## letterpress photo query: how many words to use when searching for an image

A photo query is a stock image search. Two to four plain nouns is the
right length, and a descriptive sentence is the commonest mistake.

The search matches every word at once, so each extra word narrows the
result set toward nothing.

    RIGHT   photo="sourdough bread bakery"
    RIGHT   photo="mountain hiking trail"
    RIGHT   photo="restaurant interior dining"
    WRONG   photo="sourdough loaves cooling on a wooden counter at dawn"
    WRONG   photo="warm inviting artisanal atmosphere"

The first wrong one returns nothing at all: "sourdough bread loaves on
a wooden bakery counter" matched zero images while "sourdough bread
bakery" matched two hundred and forty. The second returns nothing
useful, because mood is not a thing anyone tags a photograph with. Name
the object, the place and the material — the words a person would use
to describe what is IN the frame, not how it feels.

Two figures on one page should not both be photographs of the same
thing. A hero showing the room and a detail showing the hands at work
is a page; two rooms is a repeat.

## letterpress image licensing: which stock photographs may be used

Not every image found by a search may be published on a business site,
and not every correctly licensed image is a suitable photograph.

It searches a library of openly licensed images and accepts only
licences that permit commercial use and modification. Where the licence
requires attribution — CC BY and CC BY-SA — the credit is rendered
under the picture automatically, so the page stays compliant without
you doing anything.

It also prefers a contemporary photograph over an archival one, which
matters more than it sounds. Asked for "restaurant interior dining
table" the first result was a nineteenth-century stereograph of a
Shaker dining hall: correctly licensed, lexically perfect, and
completely wrong for a restaurant opening this year. Museum
collections sit in the same index as stock photography and are mostly
engravings and glass plates.

If nothing suitable is found, the compiler draws a motif instead and
reports the substitution. A page never ships a broken image.
