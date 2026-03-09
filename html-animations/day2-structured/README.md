# Day 2 Structured Copy

This directory is a behavior-preserving structured copy of `../Day2.html`.

The original `Day2.html` is untouched. Future Day 2 maintenance can happen here instead of inside one monolithic file.

## Entry Point

- Open `index.html` directly in the browser.

## Editing Guide

- Shared foundations live in `styles/base.css`, `styles/layout.css`, and `styles/components.css`.
- Tokenization/BPE slide styles live in `styles/slides/tokenization.css`.
- Embedding and projection slide styles live in `styles/slides/embeddings.css`.
- Attention slide styles live in `styles/slides/attention.css`.
- Shared runtime state and constants live in `scripts/constants.js` and `scripts/state.js`.
- Generic DOM and dev helpers live in `scripts/utils/`.
- Slide registry, navigation, and MathJax integration live in `scripts/core/`.
- Slide-specific interactive logic lives in `scripts/slides/`.
- App startup wiring lives in `scripts/app.js`.

## Important Rule

Do not casually rename DOM IDs or classes used by JavaScript. The structured copy preserves the original selector contracts from `Day2.html`, especially for:

- `slide-16`, `slide-18`, `slide-19`
- `projectionCanvas16`, `projectionToolbar16`
- `attn18-stage`, `attn18-overlay`
- `attn19-stage`, `attn19-cols`, `attn19-overlay`

## Refactor Intent

This first pass is structural only:

- No changes were made to the original Day 2 source file.
- The slide HTML remains in one file for safety.
- CSS and JavaScript were extracted into multiple files so the deck is easier to follow and extend.
