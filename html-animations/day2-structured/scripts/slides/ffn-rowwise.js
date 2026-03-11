const FFN28_MAX_STEP = 4;
const FFN28_CLASSES = [
  'ffn28-show-attn',
  'ffn28-show-focus',
  'ffn28-show-repeat',
  'ffn28-show-bottom'
];
const FFN28_TAKEAWAYS = [
  'Attention is the cross-token step. The FFN is the next, row-wise step.',
  'After attention, each token row already contains context from other tokens.',
  'The FFN now transforms that one row with a small neural network, without consulting other rows again.',
  'The exact same FFN is reused independently at every position.',
  'Attention mixes tokens. The FFN then non-linearly transforms each token row independently.'
];

function setFfnRowwiseStep(step) {
  const slide = document.getElementById('slide-28');
  const takeaway = document.getElementById('ffn28-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(FFN28_MAX_STEP, step));
  ffnRowwiseState.step = clamped;
  FFN28_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.textContent = FFN28_TAKEAWAYS[clamped] || FFN28_TAKEAWAYS[0];
}

function initFfnRowwiseSlide() {
  ffnRowwiseState.initialized = true;
  setFfnRowwiseStep(ffnRowwiseState.step || 0);
}

function runFfnRowwiseStep() {
  if (!ffnRowwiseState.initialized) initFfnRowwiseSlide();
  if (ffnRowwiseState.step >= FFN28_MAX_STEP) return false;
  setFfnRowwiseStep(ffnRowwiseState.step + 1);
  return true;
}

function resetFfnRowwiseSlide() {
  setFfnRowwiseStep(0);
}
