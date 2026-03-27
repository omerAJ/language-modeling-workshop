/* Slide 26 — Designing Positional Encoding
   Replace-mode teaching flow with one visible state at a time. */

const POS26_MAX_STEP = 5;
const POS26_TAKEAWAYS = [
  'Before choosing a formula, set the criteria: unique positions, stable meaning across context lengths, and smooth local change.',
  'Raw integer position is too large, too crude, and inconsistent across sequence lengths when normalized.',
  'Binary keeps values bounded and reveals the multiscale pattern, but the representation still changes in abrupt jumps.',
  'Sinusoidal embeddings turn that multiscale pattern into a smooth, fixed absolute position vector.',
  'Sin/cos gives one fixed absolute code; another common option is to learn the absolute code. But many attention patterns care more about relative offset than absolute slot.',
  'RoPE rotates each Q/K pair by absolute position; the dot product turns that into a relative-offset signal, so the same gap always gives the same attention score.'
];

function measurePositionSignalPanelHeight(panel) {
  if (!panel) return 0;

  const panelRect = panel.getBoundingClientRect();
  const children = Array.from(panel.children);
  if (children.length === 0) return Math.ceil(panelRect.height);

  let maxBottom = 0;
  children.forEach((child) => {
    const rect = child.getBoundingClientRect();
    if (!rect.width && !rect.height) return;
    maxBottom = Math.max(maxBottom, rect.bottom - panelRect.top);
  });

  return Math.ceil(maxBottom);
}

function syncPositionSignalFrameHeight() {
  const slide = document.getElementById('slide-26');
  const frame = document.getElementById('pos26-state-frame');
  const activePanel = frame ? frame.querySelector('.pos26-state-panel.is-active') : null;
  if (!slide || !frame || !activePanel) return;

  requestAnimationFrame(() => {
    frame.style.height = measurePositionSignalPanelHeight(activePanel) + 'px';
    scheduleDeckRefresh({ reason: 'pos26-frame', typeset: false });
  });
}

function setPositionSignalStep(step) {
  const slide = document.getElementById('slide-26');
  const frame = document.getElementById('pos26-state-frame');
  const takeaway = document.getElementById('pos26-takeaway');
  if (!slide || !frame || !takeaway) return;

  const clamped = Math.max(0, Math.min(POS26_MAX_STEP, step));
  positionSignalState.step = clamped;
  slide.dataset.pos26Step = String(clamped);

  const panels = Array.from(slide.querySelectorAll('.pos26-state-panel'));
  panels.forEach((panel, idx) => {
    const active = idx === clamped;
    panel.classList.toggle('is-active', active);
    panel.setAttribute('aria-hidden', active ? 'false' : 'true');
  });

  const progressSteps = slide.querySelectorAll('.pos26-progress-step');
  progressSteps.forEach((item, idx) => {
    item.classList.toggle('is-active', idx === clamped);
    item.classList.toggle('is-complete', idx < clamped);
  });

  takeaway.innerHTML = POS26_TAKEAWAYS[clamped] || POS26_TAKEAWAYS[0];

  typesetMath(slide).then(() => {
    syncPositionSignalFrameHeight();
  });
}

function initPositionSignalSlide() {
  const slide = document.getElementById('slide-26');
  if (!slide) return;

  if (!positionSignalState.resizeBound) {
    addTrackedListener(window, 'resize', () => {
      const activeSlide = document.getElementById('slide-26');
      if (!activeSlide || !activeSlide.classList.contains('active')) return;
      syncPositionSignalFrameHeight();
    });
    positionSignalState.resizeBound = true;
  }

  positionSignalState.initialized = true;
  setPositionSignalStep(positionSignalState.step || 0);
}

function runPositionSignalStep() {
  if (positionSignalState.step >= POS26_MAX_STEP) return false;
  setPositionSignalStep(positionSignalState.step + 1);
  return true;
}

function resetPositionSignalSlide() {
  const slide = document.getElementById('slide-26');
  const frame = document.getElementById('pos26-state-frame');

  positionSignalState.initialized = false;
  positionSignalState.step = 0;

  if (slide) {
    slide.dataset.pos26Step = '0';
    slide.querySelectorAll('.pos26-state-panel').forEach((panel, idx) => {
      panel.classList.toggle('is-active', idx === 0);
      panel.setAttribute('aria-hidden', idx === 0 ? 'false' : 'true');
    });
    slide.querySelectorAll('.pos26-progress-step').forEach((item, idx) => {
      item.classList.toggle('is-active', idx === 0);
      item.classList.toggle('is-complete', false);
    });
  }

  if (frame) frame.style.removeProperty('height');

  const takeaway = document.getElementById('pos26-takeaway');
  if (takeaway) takeaway.innerHTML = POS26_TAKEAWAYS[0];
}
