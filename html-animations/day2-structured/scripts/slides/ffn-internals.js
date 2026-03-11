const FFN29_MAX_STEP = 5;
const FFN29_CLASSES = [
  'ffn29-show-input',
  'ffn29-show-expand',
  'ffn29-show-gelu',
  'ffn29-show-project',
  'ffn29-show-math'
];
const FFN29_TAKEAWAYS = [
  'The FFN starts from one contextualized token row.',
  'The first linear layer expands the row into a larger hidden dimension.',
  'GELU adds non-linearity, so the FFN can do more than a single linear transform.',
  'The second linear layer projects the hidden representation back to model width.',
  'The full FFN is now visible as a neural network: expand, apply GELU, then project back.',
  'The full FFN is a two-layer MLP with a non-linearity in between, applied independently to each token row.'
];

function setFfnInternalsStep(step) {
  const slide = document.getElementById('slide-29');
  const takeaway = document.getElementById('ffn29-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(FFN29_MAX_STEP, step));
  ffnInternalsState.step = clamped;
  FFN29_CLASSES.forEach((className, idx) => {
    slide.classList.toggle(className, clamped >= idx + 1);
  });
  takeaway.innerHTML = FFN29_TAKEAWAYS[clamped] || FFN29_TAKEAWAYS[0];
  typesetMath(slide);
}

function initFfnInternalsSlide() {
  ffnInternalsState.initialized = true;
  setFfnInternalsStep(ffnInternalsState.step || 0);
}

function runFfnInternalsStep() {
  if (!ffnInternalsState.initialized) initFfnInternalsSlide();
  if (ffnInternalsState.step >= FFN29_MAX_STEP) return false;
  setFfnInternalsStep(ffnInternalsState.step + 1);
  return true;
}

function resetFfnInternalsSlide() {
  setFfnInternalsStep(0);
}
