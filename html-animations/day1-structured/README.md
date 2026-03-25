# Day 1 Structured Deck

The working version of the Day 1 presentation. `../Day1.html` is reference only — make all future changes here.

## Quick Start

Open `day1-structured/index.html` directly in a browser. For a local server:

```bash
python3 -m http.server 8123
# http://127.0.0.1:8123/day1-structured/index.html
```

## File Layout

### Entry point
- `index.html` — all slide markup, MathJax config, ordered CSS/JS includes

### Styles
- `styles/tokens.css` — font import, root scaling, color tokens
- `styles/base.css` — reset, body, typography, highlights, hidden-content, speaker notes, spacers
- `styles/layout.css` — slide shell, cards, callouts, badges, grids, flows, nav bar
- `styles/components.css` — math blocks, title gradient line, pipeline boxes
- `styles/slides/video.css` — clip stage, overlays, playback timeline
- `styles/slides/loops.css` — inference token display visuals
- `styles/slides/one-update.css` — slides 8–14 charts, loss, backprop, forward-pass visuals
- `styles/slides/reasoning.css` — slides 22–24 reasoning/drama visuals
- `styles/slides/activities.css` — slide 16 and slides 31–32 activity visuals
- `styles/slides/scale-data.css` — slides 29–30 scale/data-mixture visuals

### Scripts
- `scripts/constants.js` — slide order, clip config, shared data tables, static strings
- `scripts/state.js` — mutable runtime state
- `scripts/core/math.js` — MathJax typesetting helper
- `scripts/core/slide-registry.js` — maps slide IDs to init/step/reset handlers
- `scripts/core/navigation.js` — `goToSlide`, next/back, progress UI, interaction stepping
- `scripts/core/events.js` — nav buttons, keyboard handling, delegated click listeners
- `scripts/slides/video-clips.js` — clip load/play/pause/seek behavior
- `scripts/slides/ingredients.js` — slide 3 ingredient reveal logic
- `scripts/slides/inference-training.js` — slides 5–6 interactive loop logic
- `scripts/slides/one-update.js` — slides 10–16 chart and NTP activity logic
- `scripts/slides/reasoning.js` — slides 22–23 reasoning/drama logic
- `scripts/slides/completion.js` — slides 31–32 completion/prompt-fix logic
- `scripts/app.js` — bootstraps the deck

## Slide Map

Navigation uses a fixed custom order, not DOM order:

`0, 1, 2, 35–50, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 8, 22, 23, 24, 16, 29, 30, 31, 32, 33`

High-level grouping:
- `0–1` — title and objectives
- `2, 35–50` — pipeline animation clip sequence
- `3–6` — ingredients, roadmap, inference, training
- `7–14` — one-update deep dive
- `8, 22–24` — objective limits and reasoning-emergence bridge
- `16` — “You Are the Language Model” activity
- `29–30` — pretraining scale and data distribution
- `31–33` — completion, prompt steering, post-training

## Where To Edit What

| Goal | Edit |
|---|---|
| Text, headings, markup | `index.html` |
| Global look | `styles/tokens.css`, `styles/base.css`, `styles/layout.css` |
| Topic-specific look | `styles/slides/*.css` |
| Navigation / key handling | `scripts/core/navigation.js`, `scripts/core/events.js` |
| Slide interaction logic | `scripts/slides/*.js` |
| Static labels, order, example data | `scripts/constants.js` |

## Validation

```bash
find day1-structured/scripts -type f -name '*.js' -print0 | xargs -0 -n1 node --check
```

Manual smoke test:
- navigation: next/back/skip, keyboard arrows, restart on last slide
- clip slides: autoplay on entry, pause/reset on leave, spacebar toggle, seek/play controls
- slide 3 ingredient reveals and reset
- slide 5 inference click/step flow
- slide 6 training phases 1–5 and reset
- slides 10–14 chart/loss rendering on entry
- slide 16 blank/concept reveal behavior
- slide 22 reasoning-pressure reveal
- slide 23 suspect selection and answer reset
- slides 31–32 completion/fix reveals and reset
