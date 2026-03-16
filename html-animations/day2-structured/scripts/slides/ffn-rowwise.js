const FFN28_MAX_STEP = 4;
const FFN28_CLASSES = [
  'ffn28-show-context',
  'ffn28-show-local',
  'ffn28-show-repeat',
  'ffn28-show-params'
];
const FFN28_TAKEAWAYS = [
  'Attention gathers context. The FFN then transforms each row with the same MLP.',
  'After attention, each row is already contextualized.',
  'The FFN now applies a small neural network to one row.',
  'FFN weights are reused independently at every position.',
  'In a standard decoder block, FFN projection weights are roughly twice the attention projection weights.'
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
  typesetMath(slide);
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
