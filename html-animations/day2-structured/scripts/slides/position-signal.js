/* Slide 26 — From Position Codes to RoPE
   Replace-mode teaching flow with one visible state at a time. */

const POS26_MAX_STEP = 5;
const POS26_TAKEAWAYS = [
  'Before choosing a position method, define what success looks like: unique slots, stable meanings across context lengths, and smooth local change.',
  'Raw integer position is too large relative to the embedding values, and length-normalized position breaks cross-sequence consistency.',
  'Binary codes fix the range problem and reveal the multirate pattern, but the representation changes too abruptly from one position to the next.',
  'Sinusoids keep that fast/slow hierarchy while making nearby positions change smoothly and predictably.',
  'Absolute position is not the whole story: attention compares tokens through dot products, so relative position should shape the comparison itself.',
  'RoPE reuses sinusoidal frequencies to rotate Q/K pairs before attention, letting relative position appear through angle while preserving vector norms.'
];

function syncPositionSignalFrameHeight() {
  const slide = document.getElementById('slide-26');
  const frame = document.getElementById('pos26-state-frame');
  const activePanel = frame ? frame.querySelector('.pos26-state-panel.is-active') : null;
  if (!slide || !frame || !activePanel) return;

  requestAnimationFrame(() => {
    frame.style.height = activePanel.offsetHeight + 'px';
    scheduleActiveSlideFit({ reason: 'pos26-frame' });
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
