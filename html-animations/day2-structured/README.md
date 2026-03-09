# Day 2 Structured Deck

This folder is the maintainable version of the Day 2 presentation.

It combines:
- the pre-attention material from `Day2.html`
- the full attention continuation from `Day2-attention-only.html`

The original source files are intentionally left untouched. Future edits should happen here unless you explicitly want to change the raw source decks.

## Purpose

This deck is still a static HTML presentation, but it is no longer a single giant file with all CSS and JS inline.

The structure is:
- `index.html` holds the slide markup and script/style includes
- `styles/` holds global deck styling plus topic-specific slide styling
- `scripts/` holds shared runtime, state, data, and slide-specific interactivity

The main goal is maintainability:
- easier to find the code for one topic
- easier to change one interactive slide without touching unrelated logic
- easier to onboard a new editor

## Canonical Sources

These files are reference material only:
- `../Day2.html`
- `../Day2-attention-only.html`

This folder is the working deck:
- `./index.html`

If you are making future Day 2 changes, treat `day2-structured/` as the canonical version.

## Quick Start

Open directly:
- `day2-structured/index.html`

For normal manual editing, opening the HTML file directly is enough.

For browser automation or iframe-based validation, use a local server:

```bash
cd /home/maincoder/Documents/inside-LLM/manimations/html-animations
python3 -m http.server 8123
```

Then open:

```text
http://127.0.0.1:8123/day2-structured/index.html
```

## Folder Layout

### Entry point

- `index.html`
  - Contains all slide HTML.
  - Contains the ordered stylesheet and script includes.
  - If you want to change wording, headings, badges, or the raw slide markup, this is the first file to edit.

### Shared styles

- `styles/base.css`
  - global variables, type rules, resets, shared primitives
- `styles/layout.css`
  - slide container, slide sizing, progress bar, navigation bar
- `styles/components.css`
  - reusable cards, badges, callouts, grids, utility components

### Topic styles

- `styles/slides/tokenization.css`
  - tokenization and BPE slides
- `styles/slides/embeddings.css`
  - embeddings, geometry, projection visuals
- `styles/slides/attention.css`
  - shared attention styling for slides 18-21 plus shared `attn19-*` scaffolding
- `styles/slides/attention-aggregate.css`
  - slide 22 only
- `styles/slides/attention-matrix.css`
  - slide 23 only
- `styles/slides/attention-multihead.css`
  - slide 24 only
- `styles/slides/attention-position.css`
  - slide 25 only

### Shared runtime

- `scripts/constants.js`
  - static data, labels, timings, vector tables, slide constants
- `scripts/state.js`
  - mutable runtime state for navigation and interactive slides
- `scripts/utils/dom.js`
  - DOM helpers such as `createEl`, `createVectorRect`, listener helpers
- `scripts/utils/dev.js`
  - debug snapshots and startup checks
- `scripts/utils/attention-math.js`
  - dot products, softmax, formatting helpers, and derived attention tables
- `scripts/core/slide-registry.js`
  - maps slide IDs to their init/step/reset handlers
- `scripts/core/navigation.js`
  - `goToSlide`, `nextWithInteractions`, hidden-content stepping, deck navigation
- `scripts/core/math.js`
  - MathJax typesetting helper
- `scripts/app.js`
  - bootstraps the deck and binds input events

### Slide-specific runtime

- `scripts/slides/projection.js`
  - slide 16 projection interaction
- `scripts/slides/attention-intro.js`
  - slide 18
- `scripts/slides/attention-p1.js`
  - slide 19
- `scripts/slides/attention-qkv-scores.js`
  - slide 20
- `scripts/slides/attention-weights.js`
  - slide 21
- `scripts/slides/attention-aggregate.js`
  - slide 22
- `scripts/slides/attention-matrix.js`
  - slide 23
- `scripts/slides/attention-multihead.js`
  - slide 24
- `scripts/slides/attention-position.js`
  - slide 25
- `scripts/slides/attention-shared.js`
  - shared attention DOM/vector helpers reused by multiple attention slides

## Slide Map

The slide IDs are part of the code contract and are used by JavaScript.

- `slide-0` to `slide-14`
  - tokenization and embedding build-up
- `slide-15`
  - intentionally absent
- `slide-16`
  - embedding projection interaction
- `slide-17`
  - bridge into attention
- `slide-18`
  - attention intro: tokens update each other
- `slide-19`
  - Step 1: project into Q/K/V roles
- `slide-20`
  - Step 2: compute scores
- `slide-21`
  - Step 3: scores to weights
- `slide-22`
  - Step 4: aggregate values and add residual
- `slide-23`
  - matrix view of attention
- `slide-24`
  - multi-head attention
- `slide-25`
  - positional embeddings

If you are trying to find a slide quickly, search `id="slide-23"` or whatever slide ID you need inside `index.html`.

## How The Deck Works

At load time:
1. `app.js` caches UI references and calls `refreshSlides()`.
2. `refreshSlides()` collects all `.slide` elements from `index.html`.
3. `slide-registry.js` attaches optional handlers to interactive slides.
4. `goToSlide(0)` activates the first slide.

For interactive slides:
- each registered slide can define `init`, `step`, and `reset`
- `nextWithInteractions()` tries the slide's `step()` first
- if the slide does not consume the action, normal slide navigation continues

This means:
- static slide copy lives in `index.html`
- interactive behavior lives in the matching `scripts/slides/*.js`
- the registration for that behavior lives in `scripts/core/slide-registry.js`

## Where To Edit What

### Change text, headings, badges, or markup

Edit:
- `index.html`

Examples:
- rename a heading
- rewrite a paragraph
- add a new card to a static slide
- write equations, symbols, dimensions, or formula-like labels using LaTeX with MathJax delimiters such as `\(...\)` or `\[...\]`

### Change colors, spacing, alignment, typography, animation styling

Edit the relevant CSS file under:
- `styles/`

Rule of thumb:
- global look -> `base.css`, `layout.css`, `components.css`
- topic-specific look -> `styles/slides/*`

### Change interaction steps or animation logic

Edit the relevant slide module under:
- `scripts/slides/*`

Examples:
- change which step reveals which element
- alter timing for score comparisons
- change how slide 23 transitions between matrix states

### Change vectors, labels, timings, or derived attention values

Edit:
- `scripts/constants.js`
- `scripts/utils/attention-math.js`

Use `constants.js` for raw input data and timing constants.
Use `attention-math.js` for computed tables or math derived from those constants.

## Important Conventions

### 1. DOM IDs and classes are API

The JS relies heavily on exact selectors such as:
- `slide-23`
- `attn19-cols`
- `attn22-agg-wrap`
- `attn24-head-grid`

Do not casually rename IDs or classes without updating all dependent JS and CSS.

### 2. `state` is shared runtime, not slide content

`scripts/state.js` stores the live runtime state for navigation and interactive slides.

Do not put static content in `state`.
Use it for:
- current step index
- timer IDs
- animation flags
- initialization guards

### 3. Constants first, derived math second

`constants.js` contains source data and placeholders for derived attention values.
`attention-math.js` computes the derived tables after constants load.

If you change vector values or dimensions, check both files.

### 4. Interactive slides must be registered

Adding a module file is not enough.
If a slide needs `init`, `step`, or `reset`, add it in:
- `scripts/core/slide-registry.js`

### 5. `index.html` is still the markup source of truth

There is no HTML partial system yet.
All slide markup still lives in one file.

That is acceptable as long as:
- CSS stays split
- JS stays split
- slide ownership is clear

### 6. Write math as LaTeX

If something is mathematically meaningful, write it as LaTeX rather than ad hoc HTML formatting.

Use:
- `\(...\)` for inline math in `index.html`
- `\[...\]` for display equations in `index.html`
- `setMathHTML(...)` or `setMathText(...)` for dynamic math inserted by JS

Examples:
- use `\(Q\)`, not plain `Q` when it is a matrix symbol
- use `\(x_{\mathrm{sat}}\)`, not `x<sub>sat</sub>`
- use `\(z_j = \frac{s_j}{\sqrt{d_k}}\)`, not mixed text plus HTML subscripts
- use `\(a_j = \frac{\exp(z_j)}{\sum_{\ell} \exp(z_{\ell})}\)` when you mean one attention weight explicitly

This deck already uses MathJax. Keep notation consistent and let MathJax render it.

Attention notation convention in this deck:
- single-query slides use row/scalar notation such as `\(s_j\)`, `\(z_j\)`, and `\(a_j\)`
- matrix-view slides use matrix notation such as `\(S\)`, `\(Z\)`, `\(A\)`, and `\(O\)`, with entries like `\(a_{ij}\)`

### 7. Keep the README in sync

If you make a significant change that affects how someone edits, navigates, or understands this deck, update this README in the same pass.

That includes changes like:
- adding or removing slides
- changing the file structure
- moving logic between files
- changing interaction ownership
- changing presentation flow in a way that affects the editing map

Keep the README streamlined, concise, and accurate.

## Common Tasks

### Edit only slide copy

1. Open `index.html`
2. Find the slide ID
3. Change the text
4. Reload the page

### Tweak one interactive slide

1. Find the slide ID in `index.html`
2. Find the matching module in `scripts/slides/`
3. Find the matching style file in `styles/slides/`
4. If you changed step logic, verify the step count still feels right

### Add a new interactive slide

1. Add the markup to `index.html`
2. Add any styles in the appropriate CSS file
3. Create a new slide module under `scripts/slides/` if needed
4. Add any new state in `scripts/state.js`
5. Add any constants in `scripts/constants.js`
6. Register the slide in `scripts/core/slide-registry.js`
7. Add the script include to `index.html`
8. Update this README if the architecture, slide map, editing flow, or file ownership changed

### Change the slide order

1. Reorder the HTML in `index.html`
2. Check `slide-registry.js`
3. Check any code that references slide IDs directly
4. If `DEV` checks matter to you, update `scripts/utils/dev.js`

## Validation

### Syntax check all JS

```bash
find day2-structured/scripts -type f -name '*.js' -print0 | xargs -0 -n1 node --check
```

### Manual smoke test

Check:
- next/back/skip navigation
- keyboard navigation
- slide 16 lens switching
- slides 18-25 stepping behavior
- restart behavior on the last slide

### Attention step counts

If attention interactions feel wrong, these are the expected step counts:
- slide 18 -> 8 steps
- slide 19 -> 8 steps
- slide 20 -> 5 steps
- slide 21 -> 3 steps
- slide 22 -> 5 steps
- slide 23 -> 14 steps
- slide 24 -> 6 steps
- slide 25 -> 3 steps

If a slide advances early or gets stuck, its `set...Step()` and `run...Step()` functions are the first places to inspect.

## Known Caveats

- The deck numbering intentionally skips `slide-15`.
- `index.html` still contains all slide markup in one place.
- Some browser automation is easier over `http://127.0.0.1` than `file://` because iframes and scripted inspection are more reliable on a local server.

## Recommended Editing Order For New Contributors

If you are new to this deck, read files in this order:
1. `index.html`
2. `scripts/core/slide-registry.js`
3. `scripts/core/navigation.js`
4. the specific slide module you want to change
5. the matching CSS file
6. `scripts/constants.js`
7. `scripts/utils/attention-math.js` if the slide uses computed values

That path gives the fastest mental model of how the presentation is assembled.
