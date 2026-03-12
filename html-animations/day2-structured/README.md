# Day 2 Structured Deck

The working version of the Day 2 presentation. `../Day2.html` and `../Day2-attention-only.html` are reference only — make all future changes here.

## Quick Start

Open `day2-structured/index.html` directly in a browser. For local server (needed for some browser automation):

```bash
python3 -m http.server 8123
# http://127.0.0.1:8123/day2-structured/index.html
```

## File Layout

### Entry point
- `index.html` — all slide markup, ordered CSS/JS includes

### Styles
- `styles/tokens.css` — design tokens: typography, spacing, radius, semantic color variables
- `styles/base.css` — CSS variables, type rules, resets, utility classes
- `styles/layout.css` — slide container, cards, grids, callouts, nav bar
- `styles/components.css` — arch diagram, residual bypass, token visuals

Slide-specific (one file per topic):
- `styles/slides/tokenization.css` — slides 4–13
- `styles/slides/embeddings.css` — slides 14–17
- `styles/slides/attention.css` — slides 18–21
- `styles/slides/attention-aggregate.css` — slide 22
- `styles/slides/attention-matrix.css` — slide 23
- `styles/slides/attention-multihead.css` — slide 24
- `styles/slides/attention-position.css` — slide 25
- `styles/slides/position-story.css` — slides 25–27
- `styles/slides/ffn.css` — slide 28
- `styles/slides/ffn-story.css` — slides 28–29
- `styles/slides/block-mechanics.css` — slides 30–31 (also defines `.block30-takeaway` used on 30–33)
- `styles/slides/output-head.css` — slides 32–33

### Scripts (shared)
- `scripts/constants.js` — static data, labels, timings, vector tables
- `scripts/state.js` — mutable runtime state
- `scripts/utils/dom.js` — DOM helpers (`createEl`, `createVectorRect`, etc.)
- `scripts/utils/attention-math.js` — dot products, softmax, derived attention tables
- `scripts/utils/dev.js` — debug snapshots and startup checks
- `scripts/core/slide-registry.js` — maps slide IDs to init/step/reset handlers
- `scripts/core/navigation.js` — `goToSlide`, `nextWithInteractions`, reveal stepping
- `scripts/core/math.js` — MathJax typesetting helper
- `scripts/app.js` — bootstraps deck, binds input events

### Scripts (slide-specific)
- `scripts/slides/projection.js` — slide 16
- `scripts/slides/attention-intro.js` — slide 18
- `scripts/slides/attention-p1.js` — slide 19
- `scripts/slides/attention-qkv-scores.js` — slide 20
- `scripts/slides/attention-weights.js` — slide 21
- `scripts/slides/attention-aggregate.js` — slide 22
- `scripts/slides/attention-matrix.js` — slide 23
- `scripts/slides/attention-multihead.js` — slide 24
- `scripts/slides/order-problem.js` — slide 25
- `scripts/slides/position-signal.js` — slide 26
- `scripts/slides/position-superposition.js` — slide 27
- `scripts/slides/ffn-rowwise.js` — slide 28
- `scripts/slides/ffn-internals.js` — slide 29
- `scripts/slides/gpt-block.js` — slide 30
- `scripts/slides/attention-shared.js` — shared attention DOM/vector helpers

## Slide Map

Slide IDs are part of the code contract. To jump to one: search `id="slide-23"` in `index.html`.

| ID | Content |
|---|---|
| 0–14 | Tokenization and embedding build-up |
| 15 | Intentionally absent |
| 16 | Embedding projection (interactive) |
| 17 | Bridge into attention |
| 18 | Attention intro: tokens update each other |
| 19 | Step 1: project into Q/K/V |
| 20 | Step 2: compute scores |
| 21 | Step 3: scores to weights |
| 22 | Step 4: aggregate values + residual |
| 23 | Matrix view of attention |
| 24 | Multi-head attention |
| 25 | Order problem: attention is position-blind |
| 26 | Add position: \(x_i + p_i\) before attention |
| 27 | Why the sum still works |
| 28 | Why FFN comes after attention |
| 29 | Inside one FFN: expand → GELU → project |
| 30 | The block — walk through the slide-3 diagram now that every piece is understood |
| 31 | Stack the block L times — same shape, richer representation |
| 32 | Reading the prediction: LM head, logits, vocab softmax |
| 33 | Generate, append, repeat: autoregressive loop + decoding strategies |

## Where To Edit What

| Goal | Edit |
|---|---|
| Text, headings, markup | `index.html` |
| Global look (colors, spacing, type) | `styles/tokens.css`, `styles/base.css`, `styles/layout.css` |
| Per-topic look | `styles/slides/<topic>.css` |
| Interaction steps / animation logic | `scripts/slides/<slide>.js` |
| Vectors, labels, timings | `scripts/constants.js` |
| Derived attention values | `scripts/utils/attention-math.js` |

## Reusable Components

Before writing new CSS, check whether an existing component covers it. All of these are defined in `styles/layout.css` or `styles/components.css` and ready to use in `index.html`.

### Layout
```html
<div class="grid-2"> … </div>   <!-- 2-column grid -->
<div class="grid-3"> … </div>   <!-- 3-column grid -->
<div class="card"> … </div>     <!-- surface card with border -->
```

### Callouts
```html
<div class="callout info">    <span class="icon">ℹ️</span> <span>…</span> </div>
<div class="callout warn">    <span class="icon">⚠️</span> <span>…</span> </div>
<div class="callout success"> <span class="icon">✅</span> <span>…</span> </div>
<div class="callout question"><span class="icon">🤔</span> <span>…</span> </div>
```

### Badges
```html
<div class="badge day2">Day 2</div>
<div class="badge concept">Concept</div>
<div class="badge activity">Activity</div>
<div class="badge discussion">Discussion</div>
```

### Numbered list
```html
<ul class="obj-list">
  <li data-num="1">First point</li>
  <li data-num="2">Second point</li>
</ul>
```

### Flow diagram
```html
<div class="flow">
  <div class="flow-box">Input</div>
  <div class="flow-box">Process</div>
  <div class="flow-box">Output</div>
</div>
```
Arrows between boxes are added automatically via CSS.

### Token pills
```html
<div class="token-example">
  <span class="token-pill">word</span>         <!-- default gray -->
  <span class="token-pill main">key</span>     <!-- blue -->
  <span class="token-pill reuse">sub</span>    <!-- green -->
  <span class="token-pill flex">full</span>    <!-- orange -->
</div>
```

### Highlights (inline)
```html
<span class="hl">blue</span>  <span class="hl-green">green</span>
<span class="hl-orange">orange</span>  <span class="hl-purple">purple</span>
<span class="hl-red">red</span>  <span class="hl-cyan">cyan</span>
```

### Utility classes (from `styles/base.css`)
```
.mt-1 / .mt-2 / .mt-3 / .mt-4       margin-top (sp-1 … sp-4)
.mb-2 / .mb-3                         margin-bottom
.gap-1 / .gap-2 / .gap-3 / .gap-4    gap (sp-1 … sp-4)
.text-sm / .text-xs                   font-size (fs-sm / fs-xs)
.mono                                 JetBrains Mono font
.small-text                           dim secondary text (fs-sm)
.tiny                                 dim secondary text (fs-xs)
```

### Progressive reveal (autostep)
Click/spacebar reveals hidden elements one at a time:
```html
<div class="hidden-content autostep">…revealed on first press…</div>
<div class="hidden-content autostep">…revealed on second press…</div>
```
Navigation calls `classList.add('revealed')` then `'settled'` on each in DOM order. No JS needed for static reveals — JS is only required for animated/stateful slides.

## Design System

All visual tokens live in `styles/tokens.css`. Always use these — never hardcode raw rem values or rgba colors.

- **Typography:** `--fs-2xl` (2.8rem) → `--fs-xl` → `--fs-lg` → `--fs-base` (1.05rem) → `--fs-sm` → `--fs-xs` → `--fs-2xs` (0.72rem, diagram label floor)
- **Spacing:** `--sp-1` (0.25rem) → `--sp-2` → `--sp-3` → `--sp-4` (1.25rem) → `--sp-5` → `--sp-6`
- **Radius:** `--radius-sm` (0.375rem) · `--radius-md` (0.625rem) · `--radius-lg` (1rem) · `--radius-full`
- **Color pairs:** `--accent-bg` / `--accent-border`, same pattern for `green`, `red`, `orange`, `purple`, `cyan`, `yellow`
- **Palette:** `--accent` (blue) · `--green` · `--red` · `--orange` · `--purple` · `--yellow` · `--cyan` · `--pink`

## Conventions

**IDs and classes are API.** JS queries exact selectors like `attn19-cols`, `attn22-agg-wrap`, `attn24-head-grid`. Never rename them without updating all dependent JS and CSS.

**Interactive slides must be registered.** Creating a module file is not enough — add `init`/`step`/`reset` handlers to `scripts/core/slide-registry.js`.

**Write math as LaTeX.** Use `\(...\)` for inline, `\[...\]` for display equations. For JS-inserted math use `setMathHTML()`/`setMathText()`. Never use `<sub>` or ad hoc HTML for mathematical notation.

Attention notation convention: single-query slides use `\(s_j\)`, `\(z_j\)`, `\(a_j\)`; matrix-view slides use `\(S\)`, `\(Z\)`, `\(A\)`, `\(O\)`.

**`state.js` is runtime only.** Use it for step index, timer IDs, animation flags — never for slide content.

## Validation

```bash
# JS syntax check
find day2-structured/scripts -type f -name '*.js' -print0 | xargs -0 -n1 node --check
```

Manual smoke test: navigation (next/back/skip/keyboard), slide 16 lens switching, slides 18–30 stepping, slides 31–33 autostep reveals.

Expected step counts:

| Slide | Steps | Slide | Steps |
|---|---|---|---|
| 18 | 8 | 25 | 4 |
| 19 | 8 | 26 | 5 |
| 20 | 5 | 27 | 4 |
| 21 | 3 | 28 | 4 |
| 22 | 5 | 29 | 5 |
| 23 | 14 | 30 | 5 |
| 24 | 6 | — | — |

Autostep reveals: slide 31 → 3, slide 32 → 3, slide 33 → 3.

If a slide gets stuck, inspect its `set...Step()` and `run...Step()` functions first.

## Adding a New Interactive Slide

1. Add markup to `index.html`
2. Add styles to `styles/slides/<topic>.css` (reuse components above before writing new CSS)
3. Create `scripts/slides/<name>.js` with `init`, `step`, `reset`
4. Register it in `scripts/core/slide-registry.js`
5. Add the `<script>` include to `index.html`
6. Add any new state to `scripts/state.js`, constants to `scripts/constants.js`
7. Update this README
